"""Launch RoosterFly, the minirhizotron root annotation studio.

    python roosterfly.py --dir "path/to/image/folder"

Options: --model <run name> (a folder in segmentation/realsoil), --root <project folder holding
segmentation/realsoil> (default: this folder), --image <file>, --px-per-mm <scale>, --lookahead <n>,
--predict-all, --cache-dir <folder>.  ``python roosterfly.py --help`` lists them.
"""

from exp_realism.gui_root_annotator import main

if __name__ == "__main__":
    main()
