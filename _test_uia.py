import uiautomation as auto

fg = auto.GetForegroundControl()
print("ControlType:", fg.ControlTypeName)
print("Name:", fg.Name)
print("ClassName:", fg.ClassName)
children = fg.GetChildren()
print("Children:", len(children))
for c in children[:8]:
    name = c.Name[:60] if c.Name else "(empty)"
    ct = c.ControlTypeName
    print(f"  {ct}: {name}  class={c.ClassName}")
