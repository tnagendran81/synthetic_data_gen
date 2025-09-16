"""
Synthetic Data Generator - streaming + progress + sub-industry support

Key change: generate_full_dataset now accepts `selected_sub_industries` (list of strings).
When a set of sub-industries is provided, company/product sub_industry choices
will prefer those that match the chosen industry; otherwise generation falls back
to INDUSTRIES[industry].
"""

import os
import json
import csv
import random
import logging
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable

from faker import Faker
from dotenv import load_dotenv
load_dotenv()

# Optional OpenAI import (fallback allowed)
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Minimal consistent config used by the UI
INDUSTRIES = {
    "Technology": ["Software Development", "Cloud Computing", "Artificial Intelligence", "Cybersecurity", "IoT Solutions"],
    "Finance": ["Investment Banking", "Retail Banking", "Asset Management", "Wealth Management", "Corporate Banking"],
    "Healthcare": ["Pharmaceuticals", "Medical Devices", "Hospitals", "Telemedicine", "Biotechnology"]
}

REGIONS = {
    "APAC": {"India": "en_IN", "Singapore": "en_SG", "Australia": "en_AU"},
    "Americas": {"USA": "en_US", "Canada": "en_CA"},
    "EMEA": {"UK": "en_GB", "Germany": "en_DE"}
}

COMPANY_SIZES = {
    "startup": {"employees": (5, 100)},
    "medium": {"employees": (201, 2500)},
    "large": {"employees": (2501, 10000)}
}

MAX_EMPLOYEES_SAFE = 20000
MAX_PRODUCTS_SAFE = 30000
DEFAULT_PRODUCTS_PER_COMPANY = 3
DEFAULT_REVIEWS_PER_PRODUCT = 2

ProgressCallback = Optional[Callable[[float, str], None]]  # receives (fraction 0..1, message)


class SyntheticDataGenerator:
    def __init__(self, openai_api_key: Optional[str] = None):
        self.openai_api_key = openai_api_key
        self.openai_client = None
        if openai_api_key and OPENAI_AVAILABLE:
            try:
                self.openai_client = OpenAI(api_key=openai_api_key)
            except Exception as e:
                logger.warning(f"OpenAI init failed: {e}")

    def _safe_faker(self, locale: Optional[str]):
        try:
            if locale:
                return Faker(locale)
        except Exception:
            pass
        try:
            return Faker("en_US")
        except Exception:
            return Faker()

    def _llm_call(self, prompt: str, max_tokens: int = 100, temperature: float = 0.7) -> str:
        if self.openai_client:
            try:
                resp = self.openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                # Support different response shapes
                if hasattr(resp, "choices"):
                    return resp.choices[0].message.content.strip()
                elif isinstance(resp, dict) and "choices" in resp:
                    return resp["choices"][0]["message"]["content"].strip()
            except Exception as e:
                logger.warning(f"LLM call failed: {e}")
        return ""

    # ---------- basic generation helpers ----------
    def generate_company(self, industry: str, sub_industry: str, region: str, country: str, size: str) -> Dict[str, Any]:
        locale = REGIONS.get(region, {}).get(country, None)
        fake = self._safe_faker(locale)
        emp_range = COMPANY_SIZES.get(size, {"employees": (10, 100)})["employees"]
        return {
            "id": None,
            "name": fake.company(),
            "industry": industry,
            "sub_industry": sub_industry,
            "region": region,
            "country": country,
            "registration_number": fake.bothify(text='??######'),
            "size": size,
            "incorporation_date": fake.date_between(start_date='-30y', end_date='-1y'),
            "employee_count": random.randint(*emp_range),
            "revenue": random.randint(100000, 10000000),
            "about_us": ""
        }

    def generate_employees(self, company: Dict[str, Any], num_employees: int) -> List[Dict[str, Any]]:
        if num_employees > MAX_EMPLOYEES_SAFE:
            num_employees = MAX_EMPLOYEES_SAFE
        fake = self._safe_faker(company.get("locale"))
        employees = []
        for _ in range(num_employees):
            emp = {
                "employee_id": f"EMP{random.randint(10000,99999)}",
                "company_id": company.get("id"),
                "name": fake.name(),
                "email": fake.email(),
                "phone": fake.phone_number(),
                "job_role": fake.job(),
                "department": random.choice(["Engineering","Sales","Marketing","HR","Finance","Operations"]),
                "salary": round(random.randint(30000, 200000) * random.uniform(0.9,1.5), 2),
                "address": fake.address().replace("\n", ", "),
                "hire_date": fake.date_between(start_date=company["incorporation_date"], end_date='today')
            }
            bio = self._llm_call(f"Write a 1-line bio for a {emp['job_role']} in {emp['department']}.", max_tokens=40, temperature=0.6)
            emp["bio"] = bio or f"{emp['name']} works as a {emp['job_role']} in {emp['department']}."
            employees.append(emp)
        return employees

    def generate_product(self, company: Dict[str, Any]) -> Dict[str, Any]:
        fake = self._safe_faker(company.get("locale"))
        name = f"{company['name'].split()[0]} {fake.catch_phrase()[:30]}".strip()
        price = round(random.uniform(10, 10000), 2)
        p = {
            "product_id": f"PRD{random.randint(1000,9999)}",
            "company_id": company.get("id"),
            "name": name,
            "category": company.get("sub_industry", "General"),
            "price": price,
            "launch_date": fake.date_between(start_date=company["incorporation_date"], end_date='today'),
            "marketing_blurb": "",
            "technical_specs": {},
            "features": [],
            "reviews": [],
            "gold_label": ""
        }
        details = self._llm_call(f"Create a short blurb and 3 features for {p['name']} in {p['category']}. Return JSON with keys blurb, features, specs", max_tokens=150, temperature=0.8)
        if details:
            try:
                parsed = json.loads(details)
                p["marketing_blurb"] = parsed.get("blurb","")[:200]
                p["features"] = parsed.get("features",[]) if isinstance(parsed.get("features",[]), list) else []
                p["technical_specs"] = parsed.get("specs",{}) if isinstance(parsed.get("specs",{}), dict) else {}
            except Exception:
                p["marketing_blurb"] = details.splitlines()[0][:200]
        else:
            p["marketing_blurb"] = f"{p['name']} — reliable {p['category']} solution."
            p["features"] = ["Reliable", "Scalable", "Easy to integrate"]
            p["technical_specs"] = {"warranty":"1 year","platforms":"Cross-platform"}

        reviews_blob = self._llm_call(f"Generate 2 short reviews for product {p['name']} as JSON array (reviewer,rating,text).", max_tokens=120, temperature=0.8)
        if reviews_blob:
            try:
                p["reviews"] = json.loads(reviews_blob)
            except Exception:
                p["reviews"] = [{"reviewer":"User","rating":4,"text":"Works well."} for _ in range(DEFAULT_REVIEWS_PER_PRODUCT)]
        else:
            p["reviews"] = [{"reviewer":"User","rating":4,"text":"Works well."} for _ in range(DEFAULT_REVIEWS_PER_PRODUCT)]

        p["gold_label"] = (self._llm_call(f"Suggest one canonical label for {p['name']}.", max_tokens=20, temperature=0.2) or p["category"])
        return p

    # ---------- helper to pick sub_industry respecting UI selection ----------
    def _choose_sub_industry(self, industry: str, selected_sub_industries: Optional[List[str]]) -> str:
        """
        If selected_sub_industries is provided, prefer those that belong to the chosen `industry`.
        Otherwise fallback to a random choice from INDUSTRIES[industry].
        """
        candidates = INDUSTRIES.get(industry, [industry])
        if selected_sub_industries:
            # filter provided sub-industries that belong to this industry
            valid = [s for s in selected_sub_industries if s in candidates]
            if valid:
                return random.choice(valid)
            # if UI provided sub-industries but none match this industry, try to pick any provided
            if selected_sub_industries:
                return random.choice(selected_sub_industries)
        # fallback
        return random.choice(candidates)

    # ---------- streaming handlers (CSV / JSONL / SQLite) ----------
    # (these functions are the same as the streaming+progress implementation from before,
    #  but they now pass selected_sub_industries into sub-industry selection when generating companies/products)

    def _stream_to_csv(self, params: Dict[str, Any], output_dir: str, progress_cb: ProgressCallback = None) -> Dict[str, Any]:
        os.makedirs(output_dir, exist_ok=True)
        companies_path = os.path.join(output_dir, "companies.csv")
        employees_path = os.path.join(output_dir, "employees.csv")
        products_path = os.path.join(output_dir, "products.csv")

        selected_industries = params["selected_industries"]
        selected_regions = params["selected_regions"]
        selected_countries = params["selected_countries"]
        num_companies = params["num_companies_per_region"]
        company_sizes = params["company_sizes"]
        employee_counts = params.get("employee_counts", {}) or {}
        product_count = params.get("product_count", None)
        selected_sub_industries = params.get("selected_sub_industries", None)

        # progress estimate (heuristic)
        total_companies_est = len(selected_regions) * sum([len(selected_countries.get(r, list(REGIONS.get(r, {}).keys()))) for r in selected_regions]) * num_companies
        per_company_products = int(product_count) if product_count not in (None, "") else DEFAULT_PRODUCTS_PER_COMPANY
        est_total_products = total_companies_est * per_company_products
        est_total_employees = total_companies_est * 50
        total_steps = float(max(1, total_companies_est + est_total_employees + est_total_products))
        step = 0
        def _report(msg=""):
            nonlocal step
            step += 1
            if progress_cb:
                try:
                    progress_cb(min(1.0, step/total_steps), msg)
                except Exception:
                    pass

        comp_f = open(companies_path, "w", newline='', encoding="utf-8")
        emp_f = open(employees_path, "w", newline='', encoding="utf-8")
        prod_f = open(products_path, "w", newline='', encoding="utf-8")

        comp_writer = csv.DictWriter(comp_f, fieldnames=["id","name","industry","sub_industry","region","country","registration_number","size","incorporation_date","employee_count","revenue","about_us"])
        comp_writer.writeheader()
        emp_writer = csv.DictWriter(emp_f, fieldnames=["employee_id","company_id","name","email","phone","job_role","department","salary","address","hire_date","bio"])
        emp_writer.writeheader()
        prod_writer = csv.DictWriter(prod_f, fieldnames=["product_id","company_id","name","category","price","launch_date","marketing_blurb","technical_specs","features","reviews","gold_label"])
        prod_writer.writeheader()

        company_id = 1
        total_companies = total_employees = total_products = 0
        for region in selected_regions:
            countries = params["selected_countries"].get(region, list(REGIONS.get(region, {}).keys()))
            for country in countries:
                for _ in range(num_companies):
                    industry = random.choice(selected_industries)
                    sub_industry = self._choose_sub_industry(industry, selected_sub_industries)
                    size = random.choice(company_sizes)
                    company = self.generate_company(industry, sub_industry, region, country, size)
                    company["id"] = company_id

                    # override employee_count if provided
                    if size.lower().startswith("s") and employee_counts.get("small") not in (None, ""):
                        company["employee_count"] = int(employee_counts["small"])
                    elif size.lower().startswith("m") and employee_counts.get("medium") not in (None, ""):
                        company["employee_count"] = int(employee_counts["medium"])
                    elif size.lower().startswith("l") and employee_counts.get("large") not in (None, ""):
                        company["employee_count"] = int(employee_counts["large"])

                    company["about_us"] = self._llm_call(f"Short About Us for {company['name']} in {company['sub_industry']}.", max_tokens=80) or ""
                    comp_row = {k: company.get(k,"") for k in ["id","name","industry","sub_industry","region","country","registration_number","size","incorporation_date","employee_count","revenue","about_us"]}
                    if isinstance(comp_row["incorporation_date"], datetime):
                        comp_row["incorporation_date"] = comp_row["incorporation_date"].isoformat()
                    comp_writer.writerow(comp_row)
                    total_companies += 1
                    _report(f"Wrote company {company_id}")

                    # employees
                    num_emp = int(company["employee_count"])
                    if num_emp > MAX_EMPLOYEES_SAFE:
                        num_emp = MAX_EMPLOYEES_SAFE
                    for emp in self.generate_employees(company, num_emp):
                        emp["company_id"] = company_id
                        emp_row = {k: emp.get(k,"") for k in ["employee_id","company_id","name","email","phone","job_role","department","salary","address","hire_date","bio"]}
                        if isinstance(emp_row["hire_date"], datetime):
                            emp_row["hire_date"] = emp_row["hire_date"].isoformat()
                        emp_writer.writerow(emp_row)
                        total_employees += 1
                        _report(f"Wrote employee for company {company_id}")

                    # products
                    per_company_products = int(product_count) if product_count not in (None, "") else DEFAULT_PRODUCTS_PER_COMPANY
                    if per_company_products > MAX_PRODUCTS_SAFE:
                        per_company_products = MAX_PRODUCTS_SAFE
                    for _ in range(per_company_products):
                        p = self.generate_product(company)
                        p["company_id"] = company_id
                        prod_row = {
                            "product_id": p.get("product_id",""),
                            "company_id": p.get("company_id",""),
                            "name": p.get("name",""),
                            "category": p.get("category",""),
                            "price": p.get("price",0.0),
                            "launch_date": p.get("launch_date").isoformat() if isinstance(p.get("launch_date"), datetime) else str(p.get("launch_date","")),
                            "marketing_blurb": p.get("marketing_blurb",""),
                            "technical_specs": json.dumps(p.get("technical_specs",""), ensure_ascii=False),
                            "features": json.dumps(p.get("features",""), ensure_ascii=False),
                            "reviews": json.dumps(p.get("reviews",""), ensure_ascii=False),
                            "gold_label": p.get("gold_label","")
                        }
                        prod_writer.writerow(prod_row)
                        total_products += 1
                        _report(f"Wrote product for company {company_id}")

                    company_id += 1

        comp_f.close(); emp_f.close(); prod_f.close()
        return {"companies_path": companies_path, "employees_path": employees_path, "products_path": products_path, "summary": {"total_companies": total_companies, "total_employees": total_employees, "total_products": total_products}}

    def _stream_to_jsonl(self, params: Dict[str, Any], output_dir: str, progress_cb: ProgressCallback = None) -> Dict[str, Any]:
        os.makedirs(output_dir, exist_ok=True)
        companies_path = os.path.join(output_dir, "companies.jsonl")
        employees_path = os.path.join(output_dir, "employees.jsonl")
        products_path = os.path.join(output_dir, "products.jsonl")
        combined_json = os.path.join(output_dir, "synthetic_data.json")

        selected_industries = params["selected_industries"]
        selected_regions = params["selected_regions"]
        selected_countries = params["selected_countries"]
        num_companies = params["num_companies_per_region"]
        company_sizes = params["company_sizes"]
        employee_counts = params.get("employee_counts", {}) or {}
        product_count = params.get("product_count", None)
        selected_sub_industries = params.get("selected_sub_industries", None)

        total_companies_est = len(selected_regions) * sum([len(selected_countries.get(r, list(REGIONS.get(r, {}).keys()))) for r in selected_regions]) * num_companies
        per_company_products = int(product_count) if product_count not in (None, "") else DEFAULT_PRODUCTS_PER_COMPANY
        est_total_products = total_companies_est * per_company_products
        est_total_employees = total_companies_est * 50
        total_steps = float(max(1, total_companies_est + est_total_employees + est_total_products))
        step = 0
        def _report(msg=""):
            nonlocal step
            step += 1
            if progress_cb:
                try:
                    progress_cb(min(1.0, step/total_steps), msg)
                except Exception:
                    pass

        companies_file = open(companies_path, "w", encoding="utf-8")
        employees_file = open(employees_path, "w", encoding="utf-8")
        products_file = open(products_path, "w", encoding="utf-8")

        companies_list_for_combined = []
        try:
            company_id = 1
            for region in selected_regions:
                countries = params["selected_countries"].get(region, list(REGIONS.get(region, {}).keys()))
                for country in countries:
                    for _ in range(num_companies):
                        industry = random.choice(selected_industries)
                        sub_industry = self._choose_sub_industry(industry, selected_sub_industries)
                        size = random.choice(company_sizes)
                        company = self.generate_company(industry, sub_industry, region, country, size)
                        company["id"] = company_id

                        if size.lower().startswith("s") and employee_counts.get("small") not in (None, ""):
                            company["employee_count"] = int(employee_counts["small"])
                        elif size.lower().startswith("m") and employee_counts.get("medium") not in (None, ""):
                            company["employee_count"] = int(employee_counts["medium"])
                        elif size.lower().startswith("l") and employee_counts.get("large") not in (None, ""):
                            company["employee_count"] = int(employee_counts["large"])

                        company["about_us"] = self._llm_call(f"Short About Us for {company['name']} in {company['sub_industry']}.", max_tokens=80) or ""
                        json.dump({**company, "incorporation_date": company["incorporation_date"].isoformat()}, companies_file, default=str)
                        companies_file.write("\n")
                        companies_list_for_combined.append({"id": company_id, "name": company["name"], "size": company["size"], "employee_count": company["employee_count"]})
                        _report(f"Company {company_id} written")

                        # employees
                        num_emp = int(company["employee_count"])
                        if num_emp > MAX_EMPLOYEES_SAFE:
                            num_emp = MAX_EMPLOYEES_SAFE
                        for emp in self.generate_employees(company, num_emp):
                            emp["company_id"] = company_id
                            emp_to_write = {**emp, "hire_date": emp["hire_date"].isoformat() if isinstance(emp["hire_date"], datetime) else str(emp["hire_date"])}
                            json.dump(emp_to_write, employees_file, ensure_ascii=False, default=str)
                            employees_file.write("\n")
                            _report(f"Employee for company {company_id}")

                        # products
                        per_company_products = int(product_count) if product_count not in (None, "") else DEFAULT_PRODUCTS_PER_COMPANY
                        if per_company_products > MAX_PRODUCTS_SAFE:
                            per_company_products = MAX_PRODUCTS_SAFE
                        for _ in range(per_company_products):
                            p = self.generate_product(company)
                            p["company_id"] = company_id
                            p_to_write = {**p, "launch_date": p["launch_date"].isoformat() if isinstance(p["launch_date"], datetime) else str(p["launch_date"])}
                            json.dump(p_to_write, products_file, ensure_ascii=False, default=str)
                            products_file.write("\n")
                            _report(f"Product for company {company_id}")

                        company_id += 1

        finally:
            companies_file.close()
            employees_file.close()
            products_file.close()

        combined = {
            "generated_at": datetime.utcnow().isoformat(),
            "companies_summary": companies_list_for_combined[:200],
            "files": {"companies": companies_path, "employees": employees_path, "products": products_path}
        }
        with open(combined_json, "w", encoding="utf-8") as f:
            json.dump(combined, f, indent=2, default=str)

        return {"companies_path": companies_path, "employees_path": employees_path, "products_path": products_path, "combined_json": combined_json, "summary": {"estimated_companies": company_id-1}}

    def _stream_to_sqlite(self, params: Dict[str, Any], sqlite_path: str, progress_cb: ProgressCallback = None, batch_commit: int = 1000) -> Dict[str, Any]:
        os.makedirs(os.path.dirname(sqlite_path) or ".", exist_ok=True)
        conn = sqlite3.connect(sqlite_path, timeout=60)
        cur = conn.cursor()
        cur.execute("""CREATE TABLE IF NOT EXISTS companies (id INTEGER PRIMARY KEY, name TEXT, industry TEXT, sub_industry TEXT, region TEXT, country TEXT, registration_number TEXT, size TEXT, incorporation_date TEXT, employee_count INTEGER, revenue REAL, about_us TEXT)""")
        cur.execute("""CREATE TABLE IF NOT EXISTS employees (id INTEGER PRIMARY KEY AUTOINCREMENT, employee_id TEXT, company_id INTEGER, name TEXT, email TEXT, phone TEXT, job_role TEXT, department TEXT, salary REAL, address TEXT, hire_date TEXT, bio TEXT)""")
        cur.execute("""CREATE TABLE IF NOT EXISTS products (id INTEGER PRIMARY KEY AUTOINCREMENT, product_id TEXT, company_id INTEGER, name TEXT, category TEXT, price REAL, launch_date TEXT, marketing_blurb TEXT, technical_specs TEXT, features TEXT, reviews TEXT, gold_label TEXT)""")
        conn.commit()

        selected_industries = params["selected_industries"]
        selected_regions = params["selected_regions"]
        selected_countries = params["selected_countries"]
        num_companies = params["num_companies_per_region"]
        company_sizes = params["company_sizes"]
        employee_counts = params.get("employee_counts", {}) or {}
        product_count = params.get("product_count", None)
        selected_sub_industries = params.get("selected_sub_industries", None)

        total_companies_est = len(selected_regions) * sum([len(selected_countries.get(r, list(REGIONS.get(r, {}).keys()))) for r in selected_regions]) * num_companies
        per_company_products = int(product_count) if product_count not in (None, "") else DEFAULT_PRODUCTS_PER_COMPANY
        est_total_products = total_companies_est * per_company_products
        est_total_employees = total_companies_est * 50
        total_steps = float(max(1, total_companies_est + est_total_employees + est_total_products))
        step = 0
        def _report(msg=""):
            nonlocal step
            step += 1
            if progress_cb:
                try:
                    progress_cb(min(1.0, step/total_steps), msg)
                except Exception:
                    pass

        total_companies = total_employees = total_products = 0
        company_id = 1
        for region in selected_regions:
            countries = selected_countries.get(region, list(REGIONS.get(region, {}).keys()))
            for country in countries:
                for _ in range(num_companies):
                    industry = random.choice(selected_industries)
                    sub_industry = self._choose_sub_industry(industry, selected_sub_industries)
                    size = random.choice(company_sizes)
                    company = self.generate_company(industry, sub_industry, region, country, size)
                    company["id"] = company_id

                    if size.lower().startswith("s") and employee_counts.get("small") not in (None, ""):
                        company["employee_count"] = int(employee_counts["small"])
                    elif size.lower().startswith("m") and employee_counts.get("medium") not in (None, ""):
                        company["employee_count"] = int(employee_counts["medium"])
                    elif size.lower().startswith("l") and employee_counts.get("large") not in (None, ""):
                        company["employee_count"] = int(employee_counts["large"])

                    company["about_us"] = self._llm_call(f"Short About Us for {company['name']}.", max_tokens=80) or ""
                    cur.execute("""INSERT INTO companies (id,name,industry,sub_industry,region,country,registration_number,size,incorporation_date,employee_count,revenue,about_us) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                                (company["id"], company["name"], company["industry"], company["sub_industry"], company["region"], company["country"], company["registration_number"], company["size"], company["incorporation_date"].isoformat() if isinstance(company["incorporation_date"], datetime) else str(company["incorporation_date"]), company["employee_count"], company["revenue"], company["about_us"]))
                    total_companies += 1
                    _report(f"Wrote company {company_id}")

                    # employees
                    num_emp = int(company["employee_count"])
                    if num_emp > MAX_EMPLOYEES_SAFE:
                        num_emp = MAX_EMPLOYEES_SAFE
                    for emp in self.generate_employees(company, num_emp):
                        cur.execute("""INSERT INTO employees (employee_id, company_id, name, email, phone, job_role, department, salary, address, hire_date, bio) VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                                    (emp["employee_id"], company_id, emp["name"], emp["email"], emp["phone"], emp["job_role"], emp["department"], emp["salary"], emp["address"], emp["hire_date"].isoformat() if isinstance(emp["hire_date"], datetime) else str(emp["hire_date"]), emp.get("bio","")))
                        total_employees += 1
                        if (total_employees % batch_commit) == 0:
                            conn.commit()
                        _report(f"Wrote employee for company {company_id}")

                    # products
                    per_company_products = int(product_count) if product_count not in (None, "") else DEFAULT_PRODUCTS_PER_COMPANY
                    if per_company_products > MAX_PRODUCTS_SAFE:
                        per_company_products = MAX_PRODUCTS_SAFE
                    for _ in range(per_company_products):
                        p = self.generate_product(company)
                        cur.execute("""INSERT INTO products (product_id, company_id, name, category, price, launch_date, marketing_blurb, technical_specs, features, reviews, gold_label) VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                                    (p.get("product_id",""), company_id, p.get("name",""), p.get("category",""), p.get("price",0.0), p.get("launch_date").isoformat() if isinstance(p.get("launch_date"), datetime) else str(p.get("launch_date","")), p.get("marketing_blurb",""), json.dumps(p.get("technical_specs",""), ensure_ascii=False), json.dumps(p.get("features",""), ensure_ascii=False), json.dumps(p.get("reviews",""), ensure_ascii=False), p.get("gold_label","")))
                        total_products += 1
                        if (total_products % batch_commit) == 0:
                            conn.commit()
                        _report(f"Wrote product for company {company_id}")

                    company_id += 1

        conn.commit()
        conn.close()
        return {"sqlite_path": sqlite_path, "summary": {"total_companies": total_companies, "total_employees": total_employees, "total_products": total_products}}

    # ---------- public API ----------
    def generate_full_dataset(self,
                              selected_industries: List[str],
                              selected_regions: List[str],
                              selected_countries: Dict[str, List[str]],
                              num_companies_per_region: int,
                              company_sizes: List[str],
                              employee_counts: Optional[Dict[str, Optional[int]]] = None,
                              product_count: Optional[int] = None,
                              noise_injection: bool = False,
                              reviews_per_product: int = DEFAULT_REVIEWS_PER_PRODUCT,
                              stream: Optional[str] = None,   # "csv" | "sqlite" | "json" | None
                              output_dir: str = "data/output",
                              sqlite_path: str = "data/output/synthetic_data.db",
                              progress_cb: ProgressCallback = None,
                              selected_sub_industries: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Orchestrate dataset generation. `selected_sub_industries` is a list of
        sub-industry strings chosen in the UI (optional). When provided, generated
        companies/products will prefer those sub-industries matching the chosen industry.
        """
        params = {
            "selected_industries": selected_industries,
            "selected_regions": selected_regions,
            "selected_countries": selected_countries,
            "num_companies_per_region": num_companies_per_region,
            "company_sizes": company_sizes,
            "employee_counts": employee_counts or {},
            "product_count": product_count,
            "noise_injection": noise_injection,
            "reviews_per_product": reviews_per_product,
            "selected_sub_industries": selected_sub_industries
        }

        if stream == "csv":
            return self._stream_to_csv(params, output_dir, progress_cb=progress_cb)
        elif stream == "sqlite":
            return self._stream_to_sqlite(params, sqlite_path, progress_cb=progress_cb)
        elif stream == "json":
            return self._stream_to_jsonl(params, output_dir, progress_cb=progress_cb)
        else:
            # legacy small in-memory sample (keeps previous behavior for non-streaming)
            companies = []
            employees = []
            products = []
            company_id = 1
            for region in selected_regions:
                countries = selected_countries.get(region, list(REGIONS.get(region, {}).keys()))
                for country in countries:
                    for _ in range(num_companies_per_region):
                        industry = random.choice(selected_industries)
                        sub_industry = self._choose_sub_industry(industry, selected_sub_industries)
                        size = random.choice(company_sizes)
                        company = self.generate_company(industry, sub_industry, region, country, size)
                        company["id"] = company_id
                        if size.lower().startswith("s") and (employee_counts or {}).get("small") not in (None,""):
                            company["employee_count"] = int(employee_counts.get("small"))
                        elif size.lower().startswith("m") and (employee_counts or {}).get("medium") not in (None,""):
                            company["employee_count"] = int(employee_counts.get("medium"))
                        elif size.lower().startswith("l") and (employee_counts or {}).get("large") not in (None,""):
                            company["employee_count"] = int(employee_counts.get("large"))
                        company["about_us"] = self._llm_call(f"Short About Us for {company['name']} in {company['sub_industry']}.", max_tokens=80) or ""
                        companies.append(company)

                        emps = self.generate_employees(company, min(50, int(company["employee_count"])))
                        for e in emps:
                            e["company_id"] = company_id
                        employees.extend(emps)

                        per_company_products = product_count if product_count is not None else DEFAULT_PRODUCTS_PER_COMPANY
                        for _ in range(per_company_products):
                            p = self.generate_product(company)
                            p["company_id"] = company_id
                            products.append(p)

                        company_id += 1

            return {"companies": companies, "employees": employees, "products": products, "summary": {"total_companies": len(companies), "total_employees": len(employees), "total_products": len(products)}}
