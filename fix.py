import os

with open('app.py', 'r', encoding='utf-8') as f:
    code = f.read()

# Replace triage bug
code = code.replace(
    'v["status"] == "TRAPPED" and v["assigned_tag"] is None',
    'v["status"] in ("TRAPPED", "TRIAGED") and v["assigned_tag"] is None'
)

# Replace red assignment threshold (aggressively triage as RED)
code = code.replace(
    'if minutes > 25:',
    'if minutes > 10:'
).replace(
    'elif minutes > 15:',
    'elif minutes > 5:'
)

# Replace dispatch sorting and grouping
dispatch_old = '''        # Get triaged victims waiting for transport - priority RED > YELLOW > GREEN
        triaged = []
        for v in state.get("victims", []):
            if v["status"] == "TRIAGED" and v["assigned_tag"] is not None:
                tag_str = v["assigned_tag"]
                priority = 0 if tag_str == "RED" else (1 if tag_str == "YELLOW" else 2)
                triaged.append((priority, v))
        
        triaged.sort(key=lambda x: x[0])'''

dispatch_new = '''        # SMART AGENT FIX: Dispatches TRAPPED directly and accounts for wait time!
        triaged = []
        for v in state.get("victims", []):
            if v["status"] in ("TRAPPED", "TRIAGED") and v["assigned_tag"] is not None:
                tag_str = v["assigned_tag"]
                priority = 0 if tag_str == "RED" else (1 if tag_str == "YELLOW" else 2)
                triaged.append((priority, -v.get("minutes_since_injury", 0), v))
        
        triaged.sort(key=lambda x: (x[0], x[1]))'''
code = code.replace(dispatch_old, dispatch_new)

# Replace victim pop in dispatch
pop_old = '''            _, victim = triaged.pop(0)'''
pop_new = '''            _, _, victim = triaged.pop(0)'''
code = code.replace(pop_old, pop_new)

# Replace SAR prioritization
sar_old = '''        trapped = [
            v for v in state.get("victims", [])
            if v["status"] == "TRAPPED"
        ]
        
        for sar in free_sar[:len(trapped)]:
            if not trapped:
                break
            victim = trapped.pop(0)'''

sar_new = '''        trapped = [
            v for v in state.get("victims", [])
            if v["status"] == "TRAPPED"
        ]
        
        # Focus on those waiting the longest
        trapped.sort(key=lambda x: x.get("minutes_since_injury", 0), reverse=True)
        
        for sar in free_sar:
            if not trapped:
                break
            victim = trapped.pop(0)'''
code = code.replace(sar_old, sar_new)

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(code)

print("Smart Agent Applied")
