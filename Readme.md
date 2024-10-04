1. Install ultralytics library
 `!pip install ultralytics`
2. `conda install ipykernel` This is for running cells in vscode

3.  Install ollama 
  `curl -fsSL https://ollama.com/install.sh | sh`

4. `ollama run llama3.2` use this for Llama 3.2

5. To use with python.
  - run `ollama serve`
  - in the conda environment make sure you have `ollama` installed. `pip install ollama`
  - `ollama pull llama3.2`
  - Now you have the model pulled and you can use it locally.