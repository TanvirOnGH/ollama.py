import requests
import os

# Ollama API Docs: <https://github.com/jmorganca/ollama/blob/main/docs/api.md>
# All durations are returned in nanoseconds.
# Compatible with ollama >=v0.1.14
# Covers all endpoints as of ollama v0.1.14

# TODO: Implement tests for all endpoints.
# TODO: Implement error handling for all endpoints.
# TODO: Show examples for all endpoints.


class Ollama:
    def __init__(self, base_url: str = "http://localhost:11434"):
        """
        Initialize the ApiClient with the specified base URL.

        Args:
        - base_url (str): The base URL of the XYZ service API. Default is 'http://localhost:11434'.
        """
        self.base_url = base_url
        self.api_url = f"{base_url}/api"

    def _post_request(self, endpoint, data):
        """
        Send a POST request to the API.

        Args:
        - endpoint (str): The API endpoint.
        - data (dict): The data to be sent in the request.

        Returns:
        Tuple[int, requests.Response]: HTTP status code and the response object.
        """
        url = f"{self.api_url}/{endpoint}"
        response = requests.post(url, json=data)
        return response.status_code, response

    def _get_request(self, endpoint):
        """
        Send a GET request to the API.

        Args:
        - endpoint (str): The API endpoint.

        Returns:
        Tuple[int, requests.Response]: HTTP status code and the response object.
        """
        url = f"{self.api_url}/{endpoint}"
        response = requests.get(url)
        return response.status_code, response

    def _delete_request(self, endpoint, data):
        """
        Send a DELETE request to the API.

        Args:
        - endpoint (str): The API endpoint.
        - data (dict): The data to be sent in the request.

        Returns:
        Tuple[int, requests.Response]: HTTP status code and the response object.
        """
        url = f"{self.api_url}/{endpoint}"
        response = requests.delete(url, json=data)
        return response.status_code, response

    def _head_request(self, endpoint):
        """
        Send a HEAD request to the API.

        Args:
        - endpoint (str): The API endpoint.

        Returns:
        bool: True if the status code is 200, False otherwise.
        """
        url = f"{self.api_url}/{endpoint}"
        response = requests.head(url)
        return response.status_code == 200

    def generate_completion(
        self,
        model,
        prompt,
        format="json",
        options=None,
        system=None,
        template=None,
        context=None,
        stream=False,
        raw=False,
    ):
        """
        Generate text completion using the specified model.

        Args:
        - model (str): The name of the model.
        - prompt (str): The input prompt for text generation.
        - format (str): The format of the output. Default is 'json'.
        - options (dict): Additional options for the generation process.
        - system (str): System prompt (Overrides Modelfile).
        - template (str): The full prompt or prompt template (Overrides Modelfile).
        - context: The context parameter returned from a previous request to /generate, this can be used to keep a short conversational memory.
        - stream (bool): Whether to stream the response.
        - raw (bool): Whether to return raw response.

        Returns:
        Tuple[int, requests.Response]: HTTP status code and the response object.
        """
        endpoint = "generate"
        options = options or {}
        allowed_options = [
            "num_keep",
            "seed",
            "num_predict",
            "top_k",
            "top_p",
            "tfs_z",
            "typical_p",
            "repeat_last_n",
            "temperature",
            "repeat_penalty",
            "presence_penalty",
            "frequency_penalty",
            "mirostat",
            "mirostat_tau",
            "mirostat_eta",
            "penalize_newline",
            "stop",
            "numa",
            "num_ctx",
            "num_batch",
            "num_gqa",
            "num_gpu",
            "main_gpu",
            "low_vram",
            "f16_kv",
            "logits_all",
            "vocab_only",
            "use_mmap",
            "use_mlock",
            "embedding_only",
            "rope_frequency_base",
            "rope_frequency_scale",
            "num_thread",
        ]
        validated_options = {
            key: options[key] for key in options if key in allowed_options
        }
        parameters = {
            "model": model,
            "prompt": prompt,
            "format": format,
            "options": validated_options,
            "system": system,
            "template": template,
            "context": context,
            "stream": stream,
            "raw": raw,
        }
        return self._post_request(endpoint, parameters)

    def generate_chat_completion(
        self,
        model,
        messages,
        format="json",
        options=None,
        template=None,
        stream=True,
    ):
        """
        Generate chat completion using the specified model.

        Args:
        - model (str): The name of the model.
        - messages (list): List of chat messages, this can be used to keep a chat memory.
        - format (str): The format of the output. Default is 'json'.
        - options (dict): Additional options for the generation process.
        - template (str): The full prompt or prompt template (Overrides Modelfile).
        - stream (bool): Whether to stream the response.

        Returns:
        Tuple[int, requests.Response]: HTTP status code and the response object.
        """
        endpoint = "chat"
        options = options or {}
        parameters = {
            "model": model,
            "messages": messages,
            "format": format,
            "options": options,
            "template": template,
            "stream": stream,
        }
        return self._post_request(endpoint, parameters)

    def create_model(self, name, modelfile=None, stream=False, path=None):
        """
        Create a new model on the server.

        Args:
        - name (str): The name of the new model.
        - modelfile (str): Contents of the Modelfile.
        - stream (bool): Whether to stream the response.
        - path (str): Path to the Modelfile

        Returns:
        Tuple[int, requests.Response]: HTTP status code and the response object.
        """
        endpoint = "create"
        parameters = {
            "name": name,
            "modelfile": modelfile,
            "stream": stream,
            "path": path,
        }
        return self._post_request(endpoint, parameters)

    def blob_exists(self, digest):
        """
        Check if a blob with the specified digest exists.

        Args:
        - digest (str): The SHA256 digest of the blob.

        Returns:
        bool: True if the blob exists, False otherwise.
        """
        endpoint = f"blobs/{digest}"
        return self._head_request(endpoint)

    def create_blob(self, digest, file_path):
        """
        Create a new blob on the server.

        Args:
        - digest (str): The expected SHA256 digest of the file.
        - file_path (str): The path to the file to be uploaded.

        Returns:
        int: HTTP status code.
        """
        endpoint = f"blobs/{digest}"
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            with open(file_path, "rb") as file:
                response = requests.post(endpoint, files={"file": file})
            return response.status_code
        except Exception as e:
            print(f"Error uploading blob: {e}")
            return 500

    def list_local_models(self):
        """
        List local models available on the server.

        Returns:
        Tuple[int, requests.Response]: HTTP status code and the response object.
        """
        endpoint = "tags"
        return self._get_request(endpoint)

    def show_model_info(self, name):
        """
        Show information about a specific model.

        Args:
        - name (str): The name of the model.

        Returns:
        Tuple[int, requests.Response]: HTTP status code and the response object.
        """
        endpoint = "show"
        parameters = {"name": name}
        return self._post_request(endpoint, parameters)

    def copy_model(self, source, destination):
        """
        Copy a model from the source path to the destination path.

        Args:
        - source (str): The source path of the model.
        - destination (str): The destination path for the model.

        Returns:
        Tuple[int, requests.Response]: HTTP status code and the response object.
        """
        endpoint = "copy"
        parameters = {"source": source, "destination": destination}
        return self._post_request(endpoint, parameters)

    def delete_model(self, name):
        """
        Delete a model with the specified name.

        Args:
        - name (str): The name of the model to be deleted.

        Returns:
        Tuple[int, requests.Response]: HTTP status code and the response object.
        """
        endpoint = "delete"
        parameters = {"name": name}
        return self._delete_request(endpoint, parameters)

    def pull_model(self, name, insecure=False, stream=True):
        """
        Pull a model from the server.

        Args:
        - name (str): The name of the model to be pulled.
        - insecure (bool): Whether to use an insecure connection.
        - stream (bool): Whether to stream the response.

        Returns:
        Union[Generator[str, None, None], Tuple[int, Union[requests.Response, None]]]:
        - If stream is True, a generator yielding response lines. Otherwise, a tuple of HTTP status code and the response object.
        """
        endpoint = "pull"
        parameters = {"name": name, "insecure": insecure, "stream": stream}
        response = requests.post(
            f"{self.api_url}/{endpoint}", json=parameters, stream=stream
        )
        if stream:
            for line in response.iter_lines():
                if line:
                    yield line
        else:
            return response.status_code, response.json()

    def push_model(self, name, insecure=False, stream=True):
        """
        Push a model to the server.

        Args:
        - name (str): The name of the model to be pushed.
        - insecure (bool): Whether to use an insecure connection.
        - stream (bool): Whether to stream the response.

        Returns:
        Union[Generator[str, None, None], Tuple[int, Union[requests.Response, None]]]:
        - If stream is True, a generator yielding response lines. Otherwise, a tuple of HTTP status code and the response object.
        """
        endpoint = "push"
        parameters = {"name": name, "insecure": insecure, "stream": stream}
        response = requests.post(
            f"{self.api_url}/{endpoint}", json=parameters, stream=stream
        )
        if stream:
            for line in response.iter_lines():
                if line:
                    yield line
        else:
            return response.status_code, response.json()

    def generate_embeddings(self, model, prompt, additional_options=None):
        """
        Generate embeddings from a model.

        Parameters:
        - model (str): The name of the model.
        - prompt (str): Text to generate embeddings for.
        - additional_options (dict): Additional model parameters.

        Returns:
        Tuple[int, Union[requests.Response, None]]: Status code and response.
        """
        endpoint = "embeddings"
        additional_options = additional_options or {}
        allowed_options = []
        validated_options = {
            key: additional_options[key]
            for key in additional_options
            if key in allowed_options
        }
        parameters = {"model": model, "prompt": prompt,
                      "options": validated_options}
        return self._post_request(endpoint, parameters)
