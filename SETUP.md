# README.md

Welcome! This guide explains how to manage two types of Python environments based on your workflow:

1. **Miniconda** — for data analytics and Jupyter notebooks  
2. **pyenv + uv** — for app development (chatbots, APIs, etc.)

It also walks you through wrapping up (exporting or cleaning) environments when you're done.

---

## 📦 Part 1: Miniconda for Data Analytics

### 1. Install Miniconda

1. Download the installer from the [official website](https://docs.conda.io/en/latest/miniconda.html).
2. Run the installer:

   ```bash
   bash Miniconda3-latest-Linux-x86_64.sh
   ```
3. Allow it to initialize your shell. After installation, restart your shell or run:

   ```bash
   # For Bash
   source ~/.bashrc

   # For Zsh
   source ~/.zshrc
   ```

### 2. Create & Use a Conda Environment

```bash
# Create the environment
conda create -n data-env python=3.10 jupyter pandas matplotlib scikit-learn seaborn

# Activate the environment
conda activate data-env

# Launch Jupyter
jupyter notebook
```

### 3. Wrap-Up & Cleanup

```bash
# Export environment
conda env export > environment.yml

# Recreate from file
conda env create -f environment.yml

# Remove the environment
conda remove -n data-env --all
```

---

## 🛠 Part 2: pyenv + uv for Development

### 1. Install pyenv

```bash
# Clone pyenv into your home directory
git clone https://github.com/pyenv/pyenv.git ~/.pyenv

# Add the following to your shell config:

# For Bash (~/.bashrc)
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc

# For Zsh (~/.zshrc)
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zshrc
echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.zshrc
echo 'eval "$(pyenv init -)"' >> ~/.zshrc
```

Then restart your shell or source the file:

```bash
# Bash
source ~/.bashrc

# Zsh
source ~/.zshrc
```

### 2. Install & Set Python Version

```bash
# Show installable versions
pyenv install --list

# Install specific version
pyenv install 3.10.13

# Set it globally or locally
pyenv global 3.10.13    # All projects
# or
pyenv local 3.10.13     # Per repo

# Verify
python --version  # Should be 3.10.13
```

### 3. Install uv

```bash
pip install uv
```

### 4. Create & Use a uv Environment

```bash
# Create a new project directory
mkdir my-project && cd my-project

# Set up a virtual environment
uv venv

# Activate it
source .venv/bin/activate

# Install packages
uv pip install fastapi openai

# Save dependencies
uv pip freeze > requirements.txt
```

### 5. Wrap-Up & Cleanup

```bash
# Freeze dependencies
uv pip freeze > requirements.txt

# Deactivate
deactivate

# Delete environment
rm -rf .venv/
```

---

## 🔄 Switching Between Environments

| Task                | Command                                      |
|---------------------|----------------------------------------------|
| Activate Conda env  | `conda activate data-env`                    |
| Deactivate Conda    | `conda deactivate`                           |
| Activate Dev env    | `cd my-project && source .venv/bin/activate` |
| Deactivate Dev env  | `deactivate`                                 |

---

## ⚙️ Optional: Server Setup via SSH

When working on a remote server over SSH, follow these extra steps:

1. **SSH into the server**  
   ```bash
   ssh user@your.server.com
   ```

2. **Clone your project repo**  
   ```bash
   git clone https://github.com/yourusername/your-project.git
   cd your-project
   ```

3. **Install dependencies** (Miniconda or pyenv + uv as above)  
   - You may need to install Miniconda or pyenv on the server first.  
   - Follow the same steps in Part 1 or Part 2 but run them on the remote shell.

4. **Activate and start**  
   ```bash
   # For data analytics
   conda activate data-env
   jupyter notebook --no-browser --port=8888

   # For development
   source .venv/bin/activate
   # Run your app, e.g.:
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```

5. **Port forwarding** (if needed)  
   ```bash
   # From your local machine
   ssh -L 8888:localhost:8888 user@your.server.com
   ```

6. **Cleanup**  
   ```bash
   deactivate  # or conda deactivate
   exit
   ```

---

Feel free to customize this README for your own workflows!
