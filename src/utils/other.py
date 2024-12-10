def print_message(message, message_type="ok"):
    """
    Print a message with a specific type: ok (green), warning (yellow), or fail (red).

    :param message: The message to print.
    :param message_type: The type of the message: "ok", "warning", or "fail".
    """
    colors = {
        "ok": "\033[92m",       # Green
        "warning": "\033[93m",  # Yellow
        "fail": "\033[91m"      # Red
    }
    end_color = "\033[0m"
    color = colors.get(message_type, "\033[92m")  # Default to green if type is unknown
    print(f"{color}{message}{end_color}")