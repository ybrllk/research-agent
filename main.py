import uuid
from graph import build_graph

app = build_graph()


def run():
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    first_turn = True

    print("Research Assistant Agent (type 'exit' to quit)\n")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in ("exit", "quit"):
            print("Goodbye.")
            break

        if not user_input:
            # Error case: invalid/empty input handled at the CLI boundary.
            print("Please enter a research topic or question.\n")
            continue

        if first_turn:
            # First turn: treat input as the research topic, fresh state.
            state_input = {
                "topic": user_input,
                "messages": [],
                "collected_data": [],
                "sources": [],
                "iteration_count": 0,
            }
            first_turn = False
        else:
            # Follow-up turn: MemorySaver already has prior state for this
            # thread_id, so we only need to pass what's new. We reset
            # iteration_count so the guard applies fresh per question, and
            # update 'topic' so Planner/Researcher reason about the new ask.
            state_input = {
                "topic": user_input,
                "iteration_count": 0,
            }

        print()  # spacing before internal logs
        result = app.invoke(state_input, config)
        print(f"\nAgent: {result['messages'][-1].content}\n")


if __name__ == "__main__":
    run()