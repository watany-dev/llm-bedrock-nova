import os
import mimetypes
from io import BytesIO
from base64 import b64encode, b64decode
from typing import Optional, List

import boto3
from PIL import Image
from pydantic import Field, field_validator

import llm

# Supported image formats for the Bedrock Converse API
BEDROCK_CONVERSE_IMAGE_FORMATS = ["png", "jpeg", "gif", "webp"]
# Mapping from MIME types to Bedrock Converse-supported document formats
MIME_TYPE_TO_BEDROCK_CONVERSE_DOCUMENT_FORMAT = {
    "application/pdf": "pdf",
    "text/csv": "csv",
    "application/msword": "doc",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "application/vnd.ms-excel": "xls",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
    "text/html": "html",
    "text/plain": "txt",
    "text/markdown": "md",
}

# This maximum size is similar to the Anthropic limit in the provided code
# If Nova supports different limits, please adjust accordingly.
NOVA_MAX_IMAGE_LONG_SIZE = 1568


@llm.hookimpl
def register_models(register):
    """
    Register Amazon Nova models with llm. You can change aliases as desired.
    """
    register(
        BedrockNova("amazon.nova-pro-v1:0"),
        aliases=("bedrock-nova-pro", "nova-pro"),
    )
    register(
        BedrockNova("amazon.nova-lite-v1:0"),
        aliases=("bedrock-nova-lite", "nova-lite"),
    )
    register(
        BedrockNova("amazon.nova-micro-v1:0"),
        aliases=("bedrock-nova-micro", "nova-micro"),
    )


class BedrockNova(llm.Model):
    """
    Model class to invoke Nova on Amazon Bedrock via the Converse API.
    """

    can_stream: bool = True

    class Options(llm.Options):
        """
        Parameters that users can optionally override.
        """

        max_tokens_to_sample: Optional[int] = Field(
            description="The maximum number of tokens to generate before stopping",
            default=4096,
        )
        bedrock_model_id: Optional[str] = Field(
            description="Bedrock modelId or ARN of base, custom, or provisioned model",
            default=None,
        )
        bedrock_attach: Optional[str] = Field(
            description="Attach the given image or document file(s) to the prompt (comma-separated if multiple).",
            default=None,
        )

        @field_validator("max_tokens_to_sample")
        def validate_length(cls, max_tokens_to_sample):
            if not (0 < max_tokens_to_sample <= 1_000_000):
                raise ValueError("max_tokens_to_sample must be in range 1-1,000,000")
            return max_tokens_to_sample

    def __init__(self, model_id):
        """
        :param model_id: The modelId for invocation on Bedrock (e.g., amazon.nova-pro-v1:0).
        """
        self.model_id = model_id

    @staticmethod
    def load_and_preprocess_image(file_path):
        """
        Load and preprocess the given image for use with the Bedrock Converse API.
        * Resize if needed.
        * Convert to a supported format if needed.
        * Return (raw_bytes, format).
        """
        with open(file_path, "rb") as fp:
            img_bytes = fp.read()

        with Image.open(BytesIO(img_bytes)) as img:
            img_format = img.format
            width, height = img.size

            # Resize if the image is larger than the maximum dimension
            if width > NOVA_MAX_IMAGE_LONG_SIZE or height > NOVA_MAX_IMAGE_LONG_SIZE:
                img.thumbnail((NOVA_MAX_IMAGE_LONG_SIZE, NOVA_MAX_IMAGE_LONG_SIZE))

            # If the image is already in a supported format and no resize was done,
            # we can keep the original bytes
            if img_format.lower() in BEDROCK_CONVERSE_IMAGE_FORMATS and img.size == (
                width,
                height,
            ):
                return img_bytes, img_format.lower()

            # Otherwise, re-export the image as PNG
            with BytesIO() as buffer:
                img.save(buffer, format="PNG")
                return buffer.getvalue(), "png"

    def image_path_to_content_block(self, file_path):
        """
        Create a Bedrock Converse content block from the given image file path.
        """
        source_bytes, file_format = self.load_and_preprocess_image(file_path)
        return {"image": {"format": file_format, "source": {"bytes": source_bytes}}}

    @staticmethod
    def sanitize_file_name(file_path):
        """
        Generate a sanitized file name that conforms to the Bedrock Converse API constraints:
        * Alphanumeric characters
        * Whitespace characters (no more than one in a row)
        * Hyphens
        * Parentheses
        * Square brackets
        * Maximum length 200
        """
        head, tail = os.path.split(file_path)
        for c in tail:
            if (
                c
                not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_()[]"
            ):
                tail = tail.replace(c, "_")
        return tail[:200] if tail else "file"

    def document_path_to_content_block(self, file_path, mime_type):
        """
        Create a Bedrock Converse content block from the given document file path.
        """
        with open(file_path, "rb") as fp:
            source_bytes = fp.read()

        return {
            "document": {
                "format": MIME_TYPE_TO_BEDROCK_CONVERSE_DOCUMENT_FORMAT[mime_type],
                "name": self.sanitize_file_name(file_path),
                "source": {"bytes": source_bytes},
            }
        }

    def prompt_to_content(self, prompt):
        """
        Convert an llm.Prompt into a list of Bedrock Converse content blocks.
        This also attaches any files if 'bedrock_attach' is specified.
        """
        content = []

        # If user wants to attach files (images/documents), parse them
        if prompt.options.bedrock_attach:
            for file_path in prompt.options.bedrock_attach.split(","):
                file_path = os.path.expanduser(file_path.strip())
                mime_type, _ = mimetypes.guess_type(file_path)
                if not mime_type:
                    raise ValueError(f"Unable to guess mime type for file: {file_path}")

                if mime_type.startswith("image/"):
                    content.append(self.image_path_to_content_block(file_path))
                elif mime_type in MIME_TYPE_TO_BEDROCK_CONVERSE_DOCUMENT_FORMAT:
                    content.append(
                        self.document_path_to_content_block(file_path, mime_type)
                    )
                else:
                    raise ValueError(f"Unsupported file type for file: {file_path}")

        # Always append the prompt text
        content.append({"text": prompt.prompt})
        return content

    def encode_bytes(self, o):
        """
        Recursively replace any 'bytes' key in the object with a base64-encoded value
        (renamed to 'bytes_b64').
        """
        if isinstance(o, list):
            return [self.encode_bytes(i) for i in o]
        elif isinstance(o, dict):
            result = {}
            for key, value in o.items():
                if key == "bytes":
                    result["bytes_b64"] = b64encode(value).decode("utf-8")
                else:
                    result[key] = self.encode_bytes(value)
            return result
        else:
            return o

    def decode_bytes(self, o):
        """
        Recursively replace any 'bytes_b64' key in the object with a base64-decoded value
        (restoring the 'bytes' key).
        """
        if isinstance(o, list):
            return [self.decode_bytes(i) for i in o]
        elif isinstance(o, dict):
            result = {}
            for key, value in o.items():
                if key == "bytes_b64":
                    result["bytes"] = b64decode(value)
                else:
                    result[key] = self.decode_bytes(value)
            return result
        else:
            return o

    def build_messages(self, prompt_content, conversation) -> List[dict]:
        """
        Convert any previous conversation data into the required message structure for
        the Bedrock Converse API, then append the current user content.
        """
        messages = []
        if conversation:
            for resp in conversation.responses:
                if resp.response_json and "bedrock_user_content" in resp.response_json:
                    user_content = self.decode_bytes(
                        resp.response_json["bedrock_user_content"]
                    )
                else:
                    # Fallback if user content is not saved in response_json
                    user_content = [{"text": resp.prompt.prompt}]

                assistant_content = [{"text": resp.text()}]

                messages.append({"role": "user", "content": user_content})
                messages.append({"role": "assistant", "content": assistant_content})

        # Append the current user message
        messages.append({"role": "user", "content": prompt_content})
        return messages

    def execute(self, prompt, stream, response, conversation):
        """
        The main execution method for llm.Model.
        This constructs Bedrock Converse request parameters, then calls
        either 'converse' or 'converse_stream'.
        """
        # If a custom bedrock_model_id is provided, use that, otherwise use the default self.model_id
        bedrock_model_id = prompt.options.bedrock_model_id or self.model_id

        # Build the prompt content and conversation
        prompt_content = self.prompt_to_content(prompt)
        messages = self.build_messages(prompt_content, conversation)

        # Preserve user content in response so it can be reused in future conversation steps
        response.response_json = {
            "bedrock_user_content": self.encode_bytes(prompt_content)
        }

        # Basic inferenceConfig; add or remove parameters as needed
        inference_config = {"maxTokens": prompt.options.max_tokens_to_sample}

        # Construct parameters for the Bedrock Converse API
        params = {
            "modelId": bedrock_model_id,
            "messages": messages,
            "inferenceConfig": inference_config,
        }

        # If a system prompt is available, add it
        if prompt.system:
            params["system"] = [{"text": prompt.system}]

        # Create Bedrock client (adjust region if necessary)
        client = boto3.client("bedrock-runtime")

        if stream:
            # Streaming response
            bedrock_response = client.converse_stream(**params)
            for event in bedrock_response["stream"]:
                ((event_type, event_content),) = event.items()
                if event_type == "contentBlockDelta":
                    text_chunk = event_content["delta"]["text"]
                    yield text_chunk
        else:
            # Non-streaming response
            bedrock_response = client.converse(**params)
            # The last message from the assistant is bedrock_response['output']['message']['content'][-1]['text']
            completion = bedrock_response["output"]["message"]["content"][-1]["text"]
            yield completion
