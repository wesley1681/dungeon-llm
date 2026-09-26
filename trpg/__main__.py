"""Launch the desktop application with ``python -m trpg``."""


def main() -> None:
    from .desktop import main as desktop_main

    desktop_main()


if __name__ == "__main__":
    main()
