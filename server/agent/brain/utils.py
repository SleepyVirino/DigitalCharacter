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

def construct_work_memory(role,text,time,place,src,dest):
    new_work_memory = {
        "role": role,
        "text": text,
        "time": time,
        "place": place,
        "src": src,
        "dest": dest,
    }
    new_work_memory["event"] = organize_work_memory(new_work_memory)
    return new_work_memory

def organize_episodic_memories(memories):
    text = ""
    for memory in memories:
        text += memory["event"]
        text += "\n"
    return text
