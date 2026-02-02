# Guide on how to set up TexStudio for editing Latexfiles

Unix / Mac: Dont know.

## Windows

* Download and install TeXLive: [https://mirror.ctan.org/systems/texlive/tlnet/install-tl-windows.exe](https://mirror.ctan.org/systems/texlive/tlnet/install-tl-windows.exe) (Installation may take multiple hours.)
* Or, as a faster/smaller alternative, install MiKTeX (it installs packages on demand): [https://miktex.org/download](https://miktex.org/download)
* Download and install TexStudio: [https://www.texstudio.org/](https://www.texstudio.org/)

If you havent already, git clone this repository to your computer.

There is a file named `master.tex`. When both have finished installing open this file with TexStudio.

In the top bar, go to **Options --> Configure TeXStudio --> Build**.
In the bottom left corner there is a checkbox for **'Show Advanced Options'** — check it.
Then replace **Build & View** with the following:

```
txs:///compile | txs:///biber | txs:///compile | txs:///view
```

You are now ready to use TexStudio.

However, i also recommend going into **Shortcuts --> Menus --> Tools** and adding an additional shortcut to **'Build View'** as `Ctrl+S`.

This will make it so whenever you save the document, it also recompiles and shows you the updated version.
