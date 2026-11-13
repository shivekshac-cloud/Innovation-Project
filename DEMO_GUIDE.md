# Demo guide: for the person presenting CleanAgent

You do not need to know Python or the project. Follow the steps in order. Do Part A **at least one day before**
the presentation, on the laptop you will present from, with internet. Part B is for the day itself.

The project page is https://github.com/shivekshac-cloud/Innovation-Project

---

## Part A. One-time setup (about 20-30 minutes, needs internet)

### A1. Install Python (skip if Python 3.10-3.13 is already installed)

1. Open https://www.python.org/downloads/ and download **Python 3.12** for Windows (64-bit installer).
2. Run it. On the first screen, **tick "Add python.exe to PATH"** (bottom of the window), then click **Install Now**.
3. When it says "Setup was successful", close it.

### A2. Download the project

1. Open https://github.com/shivekshac-cloud/Innovation-Project
2. Click the green **Code** button, then **Download ZIP**.
3. Open your Downloads folder, right-click **Innovation-Project-main.zip**, choose **Extract All...**, set the
   destination to your **Desktop**, and click **Extract**.
4. You now have a folder `Desktop\Innovation-Project-main\Innovation-Project-main`. Open the inner folder: it
   contains `setup.bat`, `run_app.bat` and `README.md`. Work only inside this folder. (Ignore the `archive` folder.)

Do not run anything from inside the ZIP without extracting it; it will not work.

### A3. Install the project

1. Double-click **setup.bat**.
2. If Windows shows **"Windows protected your PC"**, click **More info**, then **Run anyway**. (This appears for
   any downloaded script; it is expected.)
3. A black window opens and downloads the libraries. This takes 5-15 minutes. Do not close it.
4. It is finished when you see **`22 passed`** and **"Setup done"**. Press any key to close.

If it says **"Python not found"** even though you installed Python: restart the laptop and try again. If it still
fails, open *Settings → Apps → Advanced app settings → App execution aliases* and switch **off** the two
"python.exe" / "python3.exe" entries, then run `setup.bat` again.

### A4. Test run (do this once before the day)

1. Double-click **run_app.bat**. A black window opens, and after about 10 seconds your browser opens the app at
   `http://localhost:8501`. If the browser does not open, type that address into Chrome or Edge yourself.
2. **Keep the black window open** while you use the app. Closing it stops the app.
3. Try the demo script in Part C once. Note how long the house dataset takes on this laptop (around 15-30 s).
4. Close the browser tab and the black window when done.

---

## Part B. On the day

1. Plug in the charger (the laptop runs faster on power), close other heavy programs (games, many Chrome tabs).
2. Double-click **run_app.bat** about 2 minutes before you present, and wait for the browser.
3. Zoom the browser to 110-125% (Ctrl and +) so the audience can read it on a projector.
4. No internet is needed.

---

## Part C. Demo script (about 7 minutes)

**1. The problem (30 s, no clicking).** "Real Indian data is messy: money written as ₹1,20,000 or 1.2 lakh or
120k, dates where 05/03 means 5 March, Bangalore next to Bengaluru, junk like `\xc2\xa0` from bad exports,
and NA or 'undisclosed' instead of blanks. Most notebooks just delete the messy rows. Our agent fixes them."

**2. How it works (45 s).** Open the **How it works** tab. Say: "It is an agent loop: observe the data, plan steps,
act using only 14 safe tools, verify every step with guardrails, and repeat. It never runs code it wrote itself,
and every change is logged with a reason."

**3. Live demo on real data (2 min).** Go to **Live demo**.
- Dataset: **Real: Indian startup funding**. Click **Run the agent** (a few seconds).
- Point at the numbers on top: issues found, cells changed, **rows dropped: 0**.
- Open **Before → after by column**, choose **City  Location**: show `Bangalore → Bengaluru`,
  `Ahemadabad → Ahmedabad` (a misspelling), and the junk-character fixes.
- Choose **Amount in USD**: show `20,00,00,000 → 200000000` and that `undisclosed` becomes blank, not a made-up number.
- Open **Issues left**: "Entries like `Bangalore / USA` are deliberately not merged into one city. The agent leaves
  what it cannot be sure about for a human."

**4. Results on synthetic data (1.5 min).** Open **Results: synthetic data**. Say: "Here we generated data with known
errors, so every fix can be checked." Show the first chart (agent 97.5% correct cells vs 66.8% for the usual
drop-the-messy-rows approach), the **unseen formats** chart ("on formats the rules were never written for it fixes
only about two thirds: this is honest, and it is why we added the LLM mode"), and the **fairness** chart
("deleting messy rows removes 44% of small-town applicants").

**5. Results on real data (1.5 min).** Open **Results: real data**.
- Startup table: agent vs a cleaning published by a human analyst; that analyst filled 974 undisclosed amounts with
  the average ($18.4M), which invents data.
- House prices: "Cleaning alone did not improve the price model. We traced it to 74 incomplete listings; with the same
  row filter, accuracy is equal and we can price 1.6% more houses." (Being honest here earns credit.)
- CarDekho: "This was a blind test: we froze the code first. It made only safe, trivial fixes and missed units like
  kmpl. We report that as-is."

**6. Close (20 s).** "It runs offline on this i3 laptop, never deletes data silently, explains every change, and we
measured it honestly on both synthetic and real data."

**Optional, only if the file `llm_cache.json` exists in the folder:** in the left sidebar choose
**Hybrid: rules + LLM**, keep **Use recorded LLM answers** ticked, and run the startup dataset again. Say: "In hybrid
mode a small language model is asked only about values the rules could not match, like old city names. These are
recorded answers from the model, replayed so the demo needs no internet." If the file does not exist, skip this.

---

## Part D. Likely questions

- **Where is the AI?** The agent loop (observe, plan, act, verify) is the AI-agent part; the planner can be rules, an
  LLM, or both. The LLM can only choose from safe tools and its answers are checked before use.
- **Why not just use pandas?** The usual pandas recipe deletes 30-37% of rows here, and more from small towns. Fixing
  values needs Indian-format knowledge (lakh, crore, day-first dates, renamed cities).
- **The startup score is 100%, isn't that suspicious?** Yes, and we say so: the rules were improved while looking at
  that file, so it is a development score. The blind test is CarDekho, where results were modest.
- **What if the LLM gives a wrong answer?** Its answer is used only if it maps to a value that already exists in the
  column, and every step must pass the guardrails. A deliberately faulty planner damaged 15% of cells without
  guardrails and 0.9% with them.
- **What happens to values it cannot read?** They become blank and the original is kept in the change log, so nothing
  is lost and nothing is guessed.
- **Where is the data from?** Public Kaggle datasets (Indian startup funding, Bengaluru house prices, CarDekho) as
  mirrored on GitHub; copies are in `data/real/`.

## Part E. If something goes wrong

| Problem | Fix |
|---|---|
| Browser did not open | Type `http://localhost:8501` into the browser. |
| "This site can't be reached" | The black window was closed. Double-click `run_app.bat` again. |
| Black window shows `Run setup.bat first` | Do Part A3. |
| App is very slow | Use the startup dataset (small). Close other programs. Plug in the charger. |
| A red error box in the app | Press **R** in the browser to rerun, or close everything and start `run_app.bat` again. |
| Port already in use | Close all black windows, wait 10 seconds, start again. |

Pre-computed results are always available in the **Results** tabs, even if the live demo fails.
