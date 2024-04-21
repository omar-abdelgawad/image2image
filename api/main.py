from app import create_app

app = create_app()


def main() -> int:
    app.run(debug=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
