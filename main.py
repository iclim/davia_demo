from davia import Davia

app = Davia()


@app.task
def get_welcome_message(name: str = "there") -> str:
    """
    Return a personalized welcome message.
    """
    return f"Hi {name}! Welcome to a Davia app !"


if __name__ == "__main__":
    app.run()