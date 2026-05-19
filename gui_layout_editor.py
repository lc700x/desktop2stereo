# gui_layout_editor.py
# 可视化调整 gui.py 控件布局，输出 gui_layout.json
# 用法: python gui_layout_editor.py

import json, os, re, tkinter as tk
from tkinter import ttk

CONFIG_FILE = "gui_layout.json"


def _parse_grid_params(code):
    """从 gui.py 全文提取所有 self.xxx.grid(row=, column=, ...) 调用。
    包括 create_widgets 和 show_*_controls 中的动态 grid。
    返回: {widget_name: (row, col, width, columnspan)}"""
    layout = {}
    # 匹配所有 self.xxx.grid(row=R, column=C, ...) 跨行模式
    pattern = r'self\.(\w+)\.grid\s*\([^)]*?\brow\s*=\s*(\d+)[^)]*?\bcolumn\s*=\s*(\d+)[^)]*?\)'
    for m in re.finditer(pattern, code, re.DOTALL):
        wname = m.group(1)
        row = int(m.group(2))
        col = int(m.group(3))
        # columnspan
        cs_match = re.search(r'columnspan\s*=\s*(\d+)', m.group(0))
        cspan = int(cs_match.group(1)) if cs_match else 1
        # width
        w_match = re.search(rf"self\.{wname}\s*=\s*.*?\bwidth\s*=\s*(\d+)", code)
        width = int(w_match.group(1)) if w_match else None
        if wname not in layout:
            layout[wname] = (row, col, width, cspan)
    return layout


def _group_by_row(layout):
    """按 row 升序分组。"""
    groups = {}
    for wname, (row, col, w, cs) in sorted(layout.items(), key=lambda x: x[1][0]):
        key = f"Row {row}"
        if key not in groups:
            groups[key] = []
        groups[key].append(wname)
    return groups


class LayoutEditor(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("GUI Layout Editor")
        self.geometry("1400x750")

        # 解析 gui.py
        code = ""
        try:
            with open("gui.py", encoding="utf-8") as f:
                code = f.read()
        except FileNotFoundError:
            pass
        self._defaults = _parse_grid_params(code)
        print(f"[LayoutEditor] Parsed {len(self._defaults)} widgets from gui.py")

        # 分组
        self._groups = _group_by_row(self._defaults)

        # 加载已有配置
        self.layout = {}
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE) as f:
                saved = json.load(f)
            for k, v in self._defaults.items():
                if k in saved:
                    self.layout[k] = tuple(saved[k])
                else:
                    self.layout[k] = v
        else:
            self.layout = dict(self._defaults)

        self._build_ui()

    def _build_ui(self):
        # 计算最长控件名宽度
        all_names = [n for wlist in self._groups.values() for n in wlist]
        max_len = max((len(n) for n in all_names), default=30)

        left = ttk.Frame(self)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        canvas = tk.Canvas(left, width=max_len * 9 + 250)
        scrollbar = ttk.Scrollbar(left, orient=tk.VERTICAL, command=canvas.yview)
        self._scroll_frame = ttk.Frame(canvas)
        self._scroll_frame.bind("<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self._scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self._vars = {}
        for group_name in sorted(self._groups.keys(), key=lambda k: int(k.split()[-1])):
            widgets = self._groups[group_name]
            gf = ttk.LabelFrame(self._scroll_frame, text=group_name)
            gf.pack(fill=tk.X, padx=3, pady=3)
            for wname in widgets:
                if wname not in self.layout:
                    continue
                row, col, width, cspan = self.layout[wname]
                f = ttk.Frame(gf)
                f.pack(fill=tk.X, padx=2, pady=1)

                e = ttk.Entry(f, width=max_len)
                e.insert(0, wname)
                e.configure(state="readonly")
                e.pack(side=tk.LEFT)

                v_row = tk.IntVar(value=row)
                v_col = tk.IntVar(value=col)
                v_w = tk.StringVar(value=str(width) if width else "")
                v_cs = tk.IntVar(value=cspan)

                ttk.Label(f, text="R").pack(side=tk.LEFT)
                ttk.Spinbox(f, from_=0, to=30, width=3, textvariable=v_row).pack(side=tk.LEFT)
                ttk.Label(f, text="C").pack(side=tk.LEFT)
                ttk.Spinbox(f, from_=0, to=4, width=3, textvariable=v_col).pack(side=tk.LEFT)
                ttk.Label(f, text="W").pack(side=tk.LEFT)
                ttk.Entry(f, width=5, textvariable=v_w).pack(side=tk.LEFT)
                ttk.Label(f, text="CS").pack(side=tk.LEFT)
                ttk.Spinbox(f, from_=1, to=5, width=3, textvariable=v_cs).pack(side=tk.LEFT)

                self._vars[wname] = (v_row, v_col, v_w, v_cs)

        right = ttk.Frame(self)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        btn_frame = ttk.Frame(right)
        btn_frame.pack(fill=tk.X, pady=5)
        ttk.Button(btn_frame, text="Preview", command=self._refresh_preview).pack(side=tk.LEFT, padx=3)
        ttk.Button(btn_frame, text="Save", command=self._save).pack(side=tk.LEFT, padx=3)
        ttk.Button(btn_frame, text="Sync to GUI", command=self._sync_to_gui).pack(side=tk.LEFT, padx=3)
        ttk.Button(btn_frame, text="Defaults", command=self._load_defaults).pack(side=tk.LEFT, padx=3)
        ttk.Label(btn_frame, text="  R=Row C=Col W=Width CS=ColSpan").pack(side=tk.LEFT, padx=10)

        self._preview = tk.Canvas(right, bg="white")
        self._preview.pack(fill=tk.BOTH, expand=True)
        self._refresh_preview()

    def _load_defaults(self):
        self.layout = dict(self._defaults)
        for wname, (v_row, v_col, v_w, v_cs) in self._vars.items():
            row, col, width, cspan = self.layout.get(wname, (0, 0, None, 1))
            v_row.set(row)
            v_col.set(col)
            v_w.set(str(width) if width else "")
            v_cs.set(cspan)
        self._refresh_preview()

    def _save(self):
        out = {}
        for wname, (v_row, v_col, v_w, v_cs) in self._vars.items():
            w_str = v_w.get().strip()
            width = int(w_str) if w_str else None
            out[wname] = [v_row.get(), v_col.get(), width, v_cs.get()]
        with open(CONFIG_FILE, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[LayoutEditor] Saved {len(out)} widgets to {CONFIG_FILE}")

    def _sync_to_gui(self):
        """Sync row/col/columnspan from editor into gui.py grid calls."""
        self._save()
        import re
        try:
            with open("gui.py", "r", encoding="utf-8") as f:
                lines = f.readlines()
        except FileNotFoundError:
            print("[LayoutEditor] gui.py not found")
            return
        updated = 0
        for wname, (v_row, v_col, v_w, v_cs) in self._vars.items():
            new_r = v_row.get()
            new_c = v_col.get()
            new_cs = v_cs.get()
            for i, line in enumerate(lines):
                changed = False
                m = re.search(rf"(self\.{re.escape(wname)}\.grid\(.*)row=(\d+)", line)
                if m and int(m.group(2)) != new_r:
                    line = line.replace(f'row={m.group(2)}', f'row={new_r}', 1)
                    changed = True
                cm = re.search(rf"(self\.{re.escape(wname)}\.grid\(.*)column=(\d+)", line)
                if cm and int(cm.group(2)) != new_c:
                    line = line.replace(f'column={cm.group(2)}', f'column={new_c}', 1)
                    changed = True
                # columnspan: replace existing value, or insert if non-default
                if m or cm:
                    cs_match = re.search(rf"columnspan\s*=\s*(\d+)", line)
                    if cs_match and int(cs_match.group(1)) != new_cs:
                        line = line.replace(f'columnspan={cs_match.group(1)}', f'columnspan={new_cs}', 1)
                        changed = True
                    elif not cs_match and new_cs != 1:
                        line = re.sub(
                            rf'(self\.{re.escape(wname)}\.grid\([^)]*)\)',
                            rf'\1, columnspan={new_cs})',
                            line, count=1
                        )
                        changed = True
                # width: sync from editor to widget definition
                new_w_str = v_w.get().strip()
                if new_w_str and re.search(rf"self\.{re.escape(wname)}\s*=", line):
                    if re.search(r'\bwidth\s*=', line):
                        # Replace entire width=... value (may be an expression)
                        line = re.sub(r'\bwidth\s*=\s*[^,\n)]+', f'width={new_w_str}', line, count=1)
                        changed = True
                    elif line.rstrip().endswith(')'):
                        # No width yet, single-line definition — insert
                        line = re.sub(r'\)(\r?\n)?$', rf', width={new_w_str})\1', line)
                        changed = True
                    else:
                        # Multi-line definition — search ahead for closing paren
                        for j in range(i+1, len(lines)):
                            if re.search(r'\)[ \t]*$', lines[j]):
                                if re.search(r'\bwidth\s*=', lines[j]):
                                    # Already has width — replace value
                                    lines[j] = re.sub(r'\bwidth\s*=\s*[^,\n)]+', f'width={new_w_str}', lines[j], count=1)
                                else:
                                    # No width yet — insert before closing paren
                                    lines[j] = re.sub(r'\)(\r?\n)?$', rf', width={new_w_str})\1', lines[j])
                                updated += 1
                                break
                if changed:
                    lines[i] = line
                    updated += 1
        with open("gui.py", "w", encoding="utf-8") as f:
            f.writelines(lines)
        print(f"[LayoutEditor] Synced {updated} grid params to gui.py")

    def _refresh_preview(self):
        cv = self._preview
        cv.delete("all")
        for wname, (v_row, v_col, v_w, v_cs) in self._vars.items():
            w_str = v_w.get().strip()
            self.layout[wname] = (v_row.get(), v_col.get(),
                                   int(w_str) if w_str else None, v_cs.get())

        CW, CH, MX = 140, 28, 30
        max_row = max(v[0] for v in self.layout.values())
        for row in range(max_row + 2):
            y = MX + row * CH
            cv.create_line(MX, y, MX + 5 * CW, y, fill="#ddd")
            cv.create_text(MX - 10, y + CH//2, text=f"R{row}", anchor="e", font=("", 8))
        for col in range(5):
            x = MX + col * CW
            cv.create_line(x, MX, x, MX + (max_row + 2) * CH, fill="#ddd")
            cv.create_text(x + CW//2, MX - 12, text=f"C{col}", font=("", 8))

        colors = ["#e3f2fd","#fce4ec","#e8f5e9","#fff3e0","#f3e5f5",
                  "#e0f7fa","#fff8e1","#f1f8e9","#ede7f6","#efebe9"]
        for i, (wname, (row, col, _w, cspan)) in enumerate(self.layout.items()):
            x = MX + col * CW + 1
            y = MX + row * CH + 1
            w = CW * cspan - 2
            color = colors[i % len(colors)]
            cv.create_rectangle(x, y, x + w, y + CH - 2, fill=color, outline="#999")
            label = wname.replace("label_","").replace("_cb","").replace("_"," ")[:22]
            cv.create_text(x + w//2, y + CH//2 - 1, text=label, font=("", 7))


if __name__ == "__main__":
    app = LayoutEditor()
    app.mainloop()
