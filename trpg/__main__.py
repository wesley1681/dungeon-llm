"""Launch the graphical application with ``python -m trpg``."""


def main() -> None:
    from .bootstrap import main as bootstrap_main

    bootstrap_main()


if __name__ == "__main__":
    main()
