class AiError(Exception):
    """The one exception the library raises on purpose.

    Look at ``e.code`` to decide what to do:

    - ``"rate_limited"``: too many calls too quickly. Wait a minute and try again.
    - ``"bad_request"``: something in the request needs fixing (read the message).
    - ``"bad_json"``: the model's reply could not be read as data.
    - ``"unavailable"``: only raised in strict mode, when the real model cannot be reached.
    """

    def __init__(self, code, message=""):
        self.code = code
        self.message = message
        super().__init__(f"[{code}] {message}" if message else code)
