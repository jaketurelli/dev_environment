# Dev Environment Template

This is an example dev environment, with a focus on robotics development using "pinocchio", consisting of:

* A Ubuntu docker container, containing:
  * C++ development tools
  * python development tools

* VSCode configuration files, containing:
  * C++ extensions
  * python extensions
  * debugging and auto-run configurations
  * various other helpful developer extensions

## Getting started

1. Make sure you have Docker + WSL2, and Vscode + Extensions installed
    * Tutorial: <https://code.visualstudio.com/docs/devcontainers/containers>
    * Links:
        * WSL2: <https://learn.microsoft.com/en-us/windows/wsl/install>
        * Docker Desktop: <https://www.docker.com/products/docker-desktop/>
        * VSCode: <https://code.visualstudio.com/download>
        * VSCode Extension: <https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers>
2. Clone/Download this repo <https://github.com/DarrenLevine/dev_environment>
3. With docker running, open VSCode and select "File > Open Folder" and open a the folder to this cloned repo.
4. A pop up should appear in the lower right hand corner, select the "Reopen in Container" option.
5. Wait for the container to build, this will only happen once, but may take awhile.
6. Run or debug a file:
    1. Open one of the example files, such as "examples/hello_world_example.py" in file editor.
    2. Once a file is open, you can run it by using the "> Run Code" command (it's suggested you binding this to a hotkey).
    3. Once a file is open, you can debug it by ensuring your debug configuration is on "auto" and then pressing F5.
