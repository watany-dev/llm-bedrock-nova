# llm-bedrock-nova

[![PyPI](https://img.shields.io/pypi/v/llm-bedrock-nova.svg)](https://pypi.org/project/llm-bedrock-nova/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/sblakey/llm-bedrock-nova/blob/main/LICENSE)

Plugin for [LLM](https://llm.datasette.io/) adding support for Amazon's Nova models.

## New: Nova Models Support
### Nova models available on us-east-1

## Installation

Install this plugin in the same environment as LLM. From the current directory
```bash
llm install llm-bedrock-nova
```

## Configuration

You will need to specify AWS Configuration with the normal boto3 and environment variables.

For example, to use the region `us-east-1` and AWS credentials under the `personal` profile, set the environment variables

```bash
export AWS_DEFAULT_REGION=us-east-1
export AWS_PROFILE=personal
```

## Usage

This plugin adds models called `bedrock-nova`.

You can query them like this:

```bash
llm -m nova-micro "Ten great names for a new space station"
```

## Options

- `max_tokens_to_sample`, default 8_191: The maximum number of tokens to generate before stopping

Use like this:
```bash
llm -m nova-lite -o max_tokens_to_sample 20 "Sing me the alphabet"
 Here is the alphabet song:

A B C D E F G
H I J
```

## Support Models

- Amazon Nova Pro
- Amazon Nova Lite
- Amazon Nova micro