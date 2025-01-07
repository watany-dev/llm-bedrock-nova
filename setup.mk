# Makefile for setting up uv and awscli v2

# Variables
UV_VERSION = 0.5.14
INSTALL_DIR = $(HOME)/.local/bin
UV_URL = https://astral.sh/uv/$(UV_VERSION)/install.sh
AWSCLI_URL = https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip
AWSCLI_ZIP = awscliv2.zip

# Default target
.PHONY: all
all: uv awscli setup-path reload-shell

# Download and install uv
.PHONY: uv
uv:
	@echo "Installing uv..."
	curl -LsSf $(UV_URL) | sh
	curl -Lo uv $(UV_URL)
	chmod +x uv
	mkdir -p $(INSTALL_DIR)
	mv uv $(INSTALL_DIR)
	uv
	uv sync
	@echo "uv installed in $(INSTALL_DIR)"

# Download, extract, and install awscli v2
.PHONY: awscli
awscli:
	@echo "Installing awscli v2..."
	curl -Lo $(AWSCLI_ZIP) $(AWSCLI_URL)
	unzip -q $(AWSCLI_ZIP)
	./aws/install -i $(INSTALL_DIR)/aws-cli -b $(INSTALL_DIR)
	rm -rf $(AWSCLI_ZIP) aws
	@echo "awscli v2 installed in $(INSTALL_DIR)"

# Ensure PATH includes INSTALL_DIR
.PHONY: setup-path
setup-path:
	@echo "Setting up PATH..."
	if ! grep -q "$(INSTALL_DIR)" $(HOME)/.bashrc; then \
		echo "export PATH=$(INSTALL_DIR):$$PATH" >> $(HOME)/.bashrc; \
		echo "PATH updated in .bashrc"; \
	else \
		echo "PATH already set in .bashrc"; \
	fi

# Reload shell to apply changes
.PHONY: reload-shell
reload-shell:
	@echo "Reloading shell to apply PATH changes..."
	. $(HOME)/.bashrc || echo "Run 'source ~/.bashrc' to apply PATH changes manually."

# Clean up temporary files
.PHONY: clean
clean:
	@echo "Cleaning up..."
	rm -rf $(AWSCLI_ZIP) aws
	@echo "Cleanup complete"

# Verify installation
.PHONY: verify
verify:
	@echo "Verifying installations..."
	@which uv && uv --version || echo "uv not found"
	@which aws && aws --version || echo "awscli not found"