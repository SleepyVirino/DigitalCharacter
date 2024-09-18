def organize_work_memory(memory):
    text = ""
    text += "Place:\n"
    text += memory["place"]
    text += "\n\n"
    text += "Time:\n"
    text += memory["time"]
    text += "\n\n"
    text += "Event:\n"
    text += memory["src"] + " said to " + memory["dest"] + ":\n"
    text += memory["text"]
    text += "\n\n"
    return text


def organize_work_memories(memories):
    text = ""
    for memory in memories:
        if memory.get("event", None):
            text += organize_work_memory(memory)
        else:
            text += memory["event"]
        text += "\n"
    return text


def organize_episodic_memories(memories):
    text = ""
    for memory in memories:
        text += memory["event"]
        text += "\n"
    return text
