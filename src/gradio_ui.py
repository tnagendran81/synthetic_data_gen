"""
Gradio UI — Sub-industry + streaming/progress support + textual progress log

Notes:
- The generator calls progress_cb(fraction, message) frequently; we capture
  the `message` strings into `log_lines` and return them to the UI at the end.
- The progress bar is still provided via gr.Progress(); the textual log shows
  the collected messages after the run finishes (or fails).
"""

import os
import gradio as gr
import pandas as pd
import json
from datetime import datetime
from src.data_generator import SyntheticDataGenerator, INDUSTRIES, REGIONS

class SyntheticDataUI:
    def __init__(self):
        self.generator = SyntheticDataGenerator(openai_api_key=os.getenv("OPENAI_API_KEY", "") or None)

    def update_sub_industries(self, selected_industries):
        """Return CheckboxGroup update for sub-industries belonging to selected industries."""
        if not selected_industries:
            return gr.update(choices=[], value=[])
        all_subs = []
        for ind in selected_industries:
            all_subs.extend(INDUSTRIES.get(ind, []))
        # remove duplicates while preserving order
        seen = set()
        uniq = [x for x in all_subs if not (x in seen or seen.add(x))]
        return gr.update(choices=uniq, value=uniq[:3])

    def update_countries(self, selected_regions):
        if not selected_regions:
            return gr.update(choices=[], value=[])
        all_countries = []
        for r in selected_regions:
            all_countries.extend(list(REGIONS.get(r, {}).keys()))
        return gr.update(choices=all_countries, value=all_countries[:1])

    def employee_inputs_visibility(self, selected_sizes):
        vs = [gr.update(visible=False, value=None) for _ in range(3)]
        if not selected_sizes:
            return vs
        if "startup" in selected_sizes:
            vs[0] = gr.update(visible=True)
        if "medium" in selected_sizes:
            vs[1] = gr.update(visible=True)
        if "large" in selected_sizes:
            vs[2] = gr.update(visible=True)
        return vs

    def _read_csv_head(self, path: str, n: int = 10) -> pd.DataFrame:
        if not os.path.exists(path):
            return pd.DataFrame()
        try:
            return pd.read_csv(path, nrows=n)
        except Exception:
            df = pd.read_csv(path)
            return df.head(n)

    def _read_jsonl_head(self, path: str, n: int = 10) -> pd.DataFrame:
        if not os.path.exists(path):
            return pd.DataFrame()
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= n:
                    break
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
        return pd.DataFrame(rows)

    def generate_and_preview(self, industries, regions, countries, num_companies, company_sizes,
                             emp_small, emp_medium, emp_large, products_count, use_streaming, db_format,
                             noise_toggle, sub_industries, progress=gr.Progress()):
        """
        Streaming generator that yields incremental UI updates while the background
        worker performs streaming generation.

        Yields tuples matching Gradio outputs:
          (summary_text, companies_df, employees_df, products_df, progress_log)

        Note: this function uses a worker thread to run the synchronous generator
        so we can yield intermediate UI snapshots while it runs.
        """
        import threading
        import time

        # Shared state between worker and UI-yielding loop
        state = {
            "finished": False,
            "result": None,
            "error": None,
            "last_fraction": 0.0,
            "last_message": "",
            "log_lines": []
        }
        lock = threading.Lock()

        # Build employee_counts
        employee_counts = {
            "small": None if emp_small in (None, "") else int(emp_small),
            "medium": None if emp_medium in (None, "") else int(emp_medium),
            "large": None if emp_large in (None, "") else int(emp_large)
        }

        # Validate product_count
        product_count_val = None
        if products_count not in (None, ""):
            product_count_val = int(products_count)
            if product_count_val < 1:
                yield ("❌ Error: Products per company must be >= 1 if provided.", pd.DataFrame(), pd.DataFrame(),
                       pd.DataFrame(), "")
                return

        # selected_countries mapping
        selected_countries = {}
        for reg in regions or []:
            region_countries = [c for c in (countries or []) if c in REGIONS.get(reg, {})]
            selected_countries[reg] = region_countries if region_countries else list(REGIONS.get(reg, {}).keys())[:1]

        # decide stream mode and fixed file paths (we can read previews mid-run from these known paths)
        output_dir = "data/output"
        os.makedirs(output_dir, exist_ok=True)
        sqlite_path = os.path.join(output_dir, "synthetic_data_stream.db")

        if use_streaming:
            if db_format == "CSV":
                stream_mode = "csv"
                companies_path = os.path.join(output_dir, "companies.csv")
                employees_path = os.path.join(output_dir, "employees.csv")
                products_path = os.path.join(output_dir, "products.csv")
            elif db_format == "SQLite":
                stream_mode = "sqlite"
                # sqlite preview will use sqlite_path
                companies_path = employees_path = products_path = sqlite_path
            elif db_format == "JSON":
                stream_mode = "json"
                companies_path = os.path.join(output_dir, "companies.jsonl")
                employees_path = os.path.join(output_dir, "employees.jsonl")
                products_path = os.path.join(output_dir, "products.jsonl")
            else:
                stream_mode = "csv"
                companies_path = os.path.join(output_dir, "companies.csv")
                employees_path = os.path.join(output_dir, "employees.csv")
                products_path = os.path.join(output_dir, "products.csv")
        else:
            stream_mode = None
            companies_path = os.path.join(output_dir, "companies.csv")
            employees_path = os.path.join(output_dir, "employees.csv")
            products_path = os.path.join(output_dir, "products.csv")

        # progress callback used by the data generator
        def progress_cb(fraction: float, message: str = ""):
            with lock:
                try:
                    # update progress bar UI
                    progress(fraction, message)
                except Exception:
                    pass
                state["last_fraction"] = float(fraction or 0.0)
                if message:
                    state["last_message"] = str(message)
                    # append timestamped message
                    ts = datetime.utcnow().isoformat()
                    state["log_lines"].append(f"{ts} - {message}")
                    # keep log reasonably bounded
                    if len(state["log_lines"]) > 2000:
                        state["log_lines"] = state["log_lines"][-2000:]

        # Worker thread target: run the (potentially long) generation task
        def worker():
            try:
                res = self.generator.generate_full_dataset(
                    selected_industries=industries,
                    selected_regions=regions,
                    selected_countries=selected_countries,
                    num_companies_per_region=int(num_companies),
                    company_sizes=company_sizes,
                    employee_counts=employee_counts,
                    product_count=product_count_val,
                    noise_injection=bool(noise_toggle),
                    reviews_per_product=2,
                    stream=stream_mode,
                    output_dir=output_dir,
                    sqlite_path=sqlite_path,
                    progress_cb=(progress_cb if use_streaming else None),
                    selected_sub_industries=sub_industries
                )
                with lock:
                    state["result"] = res
                    state["finished"] = True
            except Exception as e:
                with lock:
                    state["error"] = str(e)
                    state["finished"] = True

        # start worker thread
        t = threading.Thread(target=worker, daemon=True)
        t.start()

        # yield loop: while worker running, yield periodic snapshots
        # We yield every 0.6s (tunable). On each yield we attempt to read available previews.
        try:
            while True:
                with lock:
                    finished = state["finished"]
                    last_frac = state["last_fraction"]
                    last_msg = state["last_message"]
                    log_copy = list(state["log_lines"])

                # Build summary text from available progress info
                summary_text = f"Progress: {int(last_frac * 100)}% - {last_msg}" if last_msg else f"Progress: {int(last_frac * 100)}%"

                # Attempt to read small previews (best-effort; files may be partial)
                try:
                    if stream_mode == "json":
                        comps_df = self._read_jsonl_head(companies_path, 10)
                        emps_df = self._read_jsonl_head(employees_path, 10)
                        prods_df = self._read_jsonl_head(products_path, 10)
                    elif stream_mode == "sqlite":
                        # try reading from sqlite file (may be being written to)
                        import sqlite3
                        conn = None
                        try:
                            conn = sqlite3.connect(sqlite_path, timeout=1)
                            comps_df = pd.read_sql_query(
                                "SELECT id,name,industry,sub_industry,country,size,employee_count FROM companies LIMIT 10",
                                conn)
                            emps_df = pd.read_sql_query(
                                "SELECT employee_id,company_id,name,email,department FROM employees LIMIT 10", conn)
                            prods_df = pd.read_sql_query(
                                "SELECT product_id,company_id,name,category,price FROM products LIMIT 10", conn)
                        except Exception:
                            comps_df = pd.DataFrame()
                            emps_df = pd.DataFrame()
                            prods_df = pd.DataFrame()
                        finally:
                            if conn:
                                conn.close()
                    else:
                        # CSV or default
                        comps_df = self._read_csv_head(companies_path, 10)
                        emps_df = self._read_csv_head(employees_path, 10)
                        prods_df = self._read_csv_head(products_path, 10)
                except Exception:
                    comps_df = pd.DataFrame()
                    emps_df = pd.DataFrame()
                    prods_df = pd.DataFrame()

                # prepare log text (last N lines)
                log_text = "\n".join(log_copy[-500:]) if log_copy else ""

                # yield an update to the UI
                yield (summary_text, comps_df, emps_df, prods_df, log_text)

                if finished:
                    break
                time.sleep(0.6)

            # final yield: prefer data from final result (if available) to give accurate summary
            with lock:
                final_result = state.get("result")
                err = state.get("error")
                final_log = list(state["log_lines"])

            if err:
                yield (f"❌ Error generating: {err}", pd.DataFrame(), pd.DataFrame(), pd.DataFrame(),
                       "\n".join(final_log[-500:]))
                return

            # if streaming, final_result contains file paths + summary
            if stream_mode == "csv":
                companies_path = final_result.get("companies_path")
                employees_path = final_result.get("employees_path")
                products_path = final_result.get("products_path")
                summary = final_result.get("summary", {})
                summary_text = (f"📊 Generated (streamed to CSV)\n- Companies: {summary.get('total_companies', '?')}\n"
                                f"- Employees: {summary.get('total_employees', '?')}\n- Products: {summary.get('total_products', '?')}\n\n"
                                f"Files:\n- {companies_path}\n- {employees_path}\n- {products_path}")
                comps_df = self._read_csv_head(companies_path, 10)
                emps_df = self._read_csv_head(employees_path, 10)
                prods_df = self._read_csv_head(products_path, 10)
                yield (summary_text, comps_df, emps_df, prods_df, "\n".join(final_log[-500:]))

            elif stream_mode == "sqlite":
                sqlite_file = final_result.get("sqlite_path", sqlite_path)
                summary = final_result.get("summary", {})
                summary_text = (
                    f"📊 Generated (streamed to SQLite)\n- Companies: {summary.get('total_companies', '?')}\n"
                    f"- Employees: {summary.get('total_employees', '?')}\n- Products: {summary.get('total_products', '?')}\n\n"
                    f"SQLite DB: {sqlite_file}")
                try:
                    import sqlite3
                    conn = sqlite3.connect(sqlite_file)
                    comps_df = pd.read_sql_query(
                        "SELECT id,name,industry,sub_industry,country,size,employee_count FROM companies LIMIT 10",
                        conn)
                    emps_df = pd.read_sql_query(
                        "SELECT employee_id,company_id,name,email,department FROM employees LIMIT 10", conn)
                    prods_df = pd.read_sql_query(
                        "SELECT product_id,company_id,name,category,price FROM products LIMIT 10", conn)
                    conn.close()
                except Exception:
                    comps_df = pd.DataFrame();
                    emps_df = pd.DataFrame();
                    prods_df = pd.DataFrame()
                yield (summary_text, comps_df, emps_df, prods_df, "\n".join(final_log[-500:]))

            elif stream_mode == "json":
                companies_path = final_result.get("companies_path")
                employees_path = final_result.get("employees_path")
                products_path = final_result.get("products_path")
                combined = final_result.get("combined_json", "")
                summary = final_result.get("summary", {})
                summary_text = (
                    f"📊 Generated (streamed to JSONL)\n- Companies (lines): {summary.get('estimated_companies', '?')}\n\n"
                    f"Files:\n- {companies_path}\n- {employees_path}\n- {products_path}\n- combined summary: {combined}")
                comps_df = self._read_jsonl_head(companies_path, 10)
                emps_df = self._read_jsonl_head(employees_path, 10)
                prods_df = self._read_jsonl_head(products_path, 10)
                yield (summary_text, comps_df, emps_df, prods_df, "\n".join(final_log[-500:]))

            else:
                # legacy in-memory result
                companies = final_result.get("companies", []) if final_result else []
                employees = final_result.get("employees", []) if final_result else []
                products = final_result.get("products", []) if final_result else []
                summary = final_result.get("summary", {}) if final_result else {}
                summary_text = (f"📊 Generated (in-memory sample)\n- Companies: {summary.get('total_companies')}\n"
                                f"- Employees: {summary.get('total_employees')}\n- Products: {summary.get('total_products')}")
                yield (summary_text, pd.DataFrame(companies).head(10), pd.DataFrame(employees).head(10),
                       pd.DataFrame(products).head(10), "\n".join(final_log[-500:]))

        except GeneratorExit:
            # generator closed by Gradio / client — attempt to flag worker to stop gracefully
            with lock:
                state["finished"] = True
            return
        except Exception as e:
            # unexpected error: yield a single error payload
            with lock:
                errlog = "\n".join(state.get("log_lines", [])[-500:])
            yield (f"❌ Error generating (UI loop): {e}", pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), errlog)
            return

    def create_interface(self):
        defaults_sizes = ["startup", "medium"]
        with gr.Blocks(title="Synthetic Data Generator — Sub-industry enabled + progress log") as app:
            gr.Markdown("# Synthetic Data Generator")
            with gr.Row():
                with gr.Column(scale=1):
                    industries = gr.CheckboxGroup(choices=list(INDUSTRIES.keys()), value=list(INDUSTRIES.keys())[:2], label="Industries")
                    sub_industries = gr.CheckboxGroup(choices=[], value=[], label="Sub-industries (optional) — auto-updates from Industries")
                    regions = gr.CheckboxGroup(choices=list(REGIONS.keys()), value=list(REGIONS.keys())[:1], label="Regions")
                    countries = gr.CheckboxGroup(choices=[], value=[], label="Countries (auto)")
                    num_companies = gr.Slider(minimum=1, maximum=50, value=1, step=1, label="Companies per Region")
                    company_sizes = gr.CheckboxGroup(choices=["startup","medium","large"], value=defaults_sizes, label="Company sizes")
                    employees_small = gr.Number(label="Employees (SMALL 5–200)", value=50, precision=0, minimum=5, maximum=200, visible=("startup" in defaults_sizes))
                    employees_medium = gr.Number(label="Employees (MEDIUM 201–2500)", value=500, precision=0, minimum=201, maximum=2500, visible=("medium" in defaults_sizes))
                    employees_large = gr.Number(label="Employees (LARGE 2501–10000)", value=5000, precision=0, minimum=2501, maximum=10000, visible=("large" in defaults_sizes))
                    products_per_company = gr.Number(label="Products per company (100–10000) - leave blank for default 3", value=None, precision=0, minimum=1, maximum=10000)
                    use_streaming = gr.Checkbox(label="Use streaming writes (recommended for large runs)", value=True)
                    db_format = gr.Dropdown(choices=["CSV","SQLite","JSON"], value="CSV", label="Output format")
                    noise_checkbox = gr.Checkbox(label="Inject Variations / Noise (typos, missing fields, duplicates)", value=False)
                    generate_btn = gr.Button("Generate")

                with gr.Column(scale=2):
                    summary = gr.Markdown("Click Generate")
                    # Add progress log textbox under preview tabs
                    with gr.Tabs():
                        with gr.TabItem("Companies"):
                            comps_table = gr.Dataframe(headers=["id","name","industry","sub_industry","country","size","employee_count"], interactive=False)
                        with gr.TabItem("Employees"):
                            emps_table = gr.Dataframe(headers=["employee_id","company_id","name","email","department"], interactive=False)
                        with gr.TabItem("Products"):
                            prods_table = gr.Dataframe(headers=["product_id","company_id","name","category","price"], interactive=False)
                    progress_log = gr.Textbox(label="Progress log (latest messages from generator)", interactive=False, lines=12)

            # events wiring
            industries.change(fn=self.update_sub_industries, inputs=[industries], outputs=[sub_industries])
            regions.change(fn=self.update_countries, inputs=[regions], outputs=[countries])
            company_sizes.change(fn=self.employee_inputs_visibility, inputs=[company_sizes], outputs=[employees_small, employees_medium, employees_large])

            # generate click; include sub_industries in inputs and add progress_log to outputs
            generate_btn.click(
                fn=self.generate_and_preview,
                inputs=[industries, regions, countries, num_companies, company_sizes,
                        employees_small, employees_medium, employees_large,
                        products_per_company, use_streaming, db_format, noise_checkbox, sub_industries],
                outputs=[summary, comps_table, emps_table, prods_table, progress_log]
            )
        return app

def launch_app():
    ui = SyntheticDataUI()
    app = ui.create_interface()
    app.launch(server_name="0.0.0.0", server_port=7860, debug=True)

if __name__ == "__main__":
    launch_app()
