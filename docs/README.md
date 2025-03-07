# RasterToolkit docs

## Tutorials

Please see the `tutorials` subfolder.

## Everything else

This folder includes source code for building the docs. Users are unlikely to need to do this themselves.

To build the docs, follow these steps:

0. Make sure pandoc is installed.

1.  Make sure dependencies are installed::
    ```
    pip install -r requirements.txt
    ```

2.  Make the documents; there are many build options. In most cases, running `./build_docs` (to rerun the tutorials; takes 2 min) or `./build_docs never` (does not rebuild the tutorials; takes 15 s) is best. Alternatively, one can call `make html` directly.

3.  The built documents will be in `./_build/html`.

## Using codespaces or vscode

1. Create a codespace and install uv and pixi
2. Install dependencies
    ```
    uv pip install -e .
    cd docs
    pixi global install pandoc
    uv pip install -r requirements.txt
    ```
3. Add [Live Server](https://marketplace.visualstudio.com/items?itemName=ritwickdey.LiveServer) extension and point to build in `.vcode/settings.json`:
    ```
    {
        "liveServer.settings.root": "/docs/_build/html"
    }    
    ```
4. Build docs follow instructions above.

## Making a pdf of the docs

In `docs/` run `make simplepdf`.
