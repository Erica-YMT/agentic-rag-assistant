class Memory:

    def __init__(self):
        self.sessions = {}

    def create_session(
        self,
        session_id
    ):
        if session_id not in self.sessions:
            self.sessions[session_id] = []

    def add_message(
        self,
        session_id,
        role,
        content
    ):
        self.create_session(
            session_id
        )
        self.sessions[session_id].append(
            {
                "role": role,
                "content": content
            }
        )

    def get_messages(
        self,
        session_id
    ):
        self.create_session(
            session_id
        )
        return self.sessions[session_id]