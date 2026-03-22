import os
from dotenv import load_dotenv
from livekit.agents import AgentSession, Agent, WorkerOptions, cli
from livekit.plugins import openai, silero
from rag import load_data, ask
import rag

load_dotenv()
load_data()


class BankAssistant(Agent):
    def __init__(self):
        super().__init__(instructions=(
            "You are a silent relay agent. "
            "You NEVER generate your own responses. "
            "The system already handles all answers automatically. "
            "Do not say anything. Do not respond. Stay completely silent."
        ))

    async def on_user_turn_completed(self, turn_ctx, new_message):
        user_question = new_message.content
        if isinstance(user_question, list):
            user_question = " ".join(
                p.text if hasattr(p, "text") else str(p)
                for p in user_question
            )
        print("AGENT USING RAG FILE:", rag.__file__)
        print("ASK FUNCTION OBJECT:", ask)
        print("USER QUESTION:", repr(user_question))
        answer = ask(user_question)
        await self.session.say(answer, allow_interruptions=True, add_to_chat_ctx=False)


async def entrypoint(ctx):
    await ctx.connect()

    session = AgentSession(
        stt=openai.STT(
            model="whisper-1",
            language="hy",
            api_key=os.getenv("OPENAI_API_KEY"),
        ),
        llm=None,
        tts=openai.TTS(
            model="tts-1",
            voice="alloy",
            api_key=os.getenv("OPENAI_API_KEY"),
        ),
        vad=silero.VAD.load()
    )

    await session.start(
        room=ctx.room,
        agent=BankAssistant(),
    )
    await session.say(
        "Բարև։ ես կարող եմ օգնել միայն վարկերի, ավանդների, և մասնաճյուղերի վերաբերյալ հարցերով։ Ի՞նչ հարց ունեք",
         allow_interruptions = True,
          add_to_chat_ctx = False,
    )


if __name__ == "__main__":
    cli.run_app(WorkerOptions(
        entrypoint_fnc=entrypoint,
        api_key=os.getenv("LIVEKIT_API_KEY"),
        api_secret=os.getenv("LIVEKIT_API_SECRET"),
        ws_url=os.getenv("LIVEKIT_URL")
    ))