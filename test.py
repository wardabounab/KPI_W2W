"""
test.py — Diagnostic rapide
Vérifie que Flask voit bien le template et les listes.
"""
import os
from jinja2 import Environment, FileSystemLoader

# 1) Le dossier templates existe ?
tpl_dir = os.path.abspath("templates")
print("Templates dir :", tpl_dir, "| exists:", os.path.isdir(tpl_dir))
if os.path.isdir(tpl_dir):
    print("Fichiers dedans :", os.listdir(tpl_dir))

# 2) Le fichier index.html est bien nommé ?
tpl_file = os.path.join(tpl_dir, "index.html")
print("index.html     :", tpl_file, "| exists:", os.path.isfile(tpl_file))

# 3) Jinja peut rendre le template avec les listes ?
from calculations.Dropdowns import WELLS, RIGS, PHASES, ROTARY_SYSTEMS
env = Environment(loader=FileSystemLoader(tpl_dir))
tpl = env.get_template("index.html")
html = tpl.render(wells=WELLS, rigs=RIGS, phases=PHASES, rotary_systems=ROTARY_SYSTEMS)

# 4) Est-ce que les valeurs apparaissent dans le HTML final ?
print()
print("=== Vérification ===")
print("TXNO-11 dans le HTML ?", "TXNO-11" in html)
print("TP 184  dans le HTML ?", "TP 184" in html)
print("Top drive dans le HTML ?", "Top drive" in html)
print("{{ w }}  dans le HTML ?", "{{ w }}" in html)   # doit être False
print("{% for   dans le HTML ?", "{% for" in html)    # doit être False