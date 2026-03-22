import os
import re
import chromadb
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

chroma_client = chromadb.PersistentClient(path="./db")
collection = chroma_client.get_or_create_collection(name="bank_data")


def load_data():
    count = collection.count()
    print(f"Database has {count} chunks loaded.")

def normalize_text(text: str) -> str:
    text = text.lower()

    # remove punctuation except Armenian/Latin letters, digits, spaces
    text = re.sub(r"[^\w\sա-ֆԱ-Ֆև]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    # phrase-level fixes first
    phrase_fixes = {
        "id bank": "իդբանկ",
        "idbank": "իդբանկ",
        "այդի բանկ": "իդբանկ",
        "այդի բանգ": "իդբանկ",
        "այդի բանք": "իդբանկ",
        "իդ բանկ": "իդբանկ",
        "որտեղ է": "որտեղ",
        "որտեղ են": "որտեղ",
        "որ տեղ": "որտեղ",
        "որ տղան": "որտեղ",
        "որտեղ կարելի է": "որտեղ կարելի է",
    }

    for wrong, correct in phrase_fixes.items():
        text = text.replace(wrong, correct)

    words = text.split()

    token_fixes = {
        # common banking typos
        "տոքոս": "տոկոս",
        "դոգոս": "տոկոս",
        "վարգ": "վարկ",
        "բանգ": "բանկ",
        "բանք": "բանկ",
        "բանքի": "բանկի",
        "բանգի": "բանկի",

        # branch typos
        "մասնաչուղ": "մասնաճյուղ",
        "մասնաչյուղ": "մասնաճյուղ",
        "մասնաչող": "մասնաճյուղ",
        "մասնակյուղ": "մասնաճյուղ",
        "մասնաչուղերը": "մասնաճյուղերը",
        "մասնաճու": "մասնաճյուղ",
        "մասնաչու": "մասնաճյուղ",

        # question words / typo noise
        "որտաղ": "որտեղ",
        "ինս": "ինձ",
        "պացել": "բացել",
        "պացեմ": "բացեմ",
        "ավանց": "ավանդ",
        "ավանցներ": "ավանդներ",
        "խնայարական": "խնայողական",
        "սպարողական": "սպառողական",

        # ameria
        "ամերյաբանգ": "ամերիաբանկ",
        "ամերյաբանք": "ամերիաբանկ",
        "ամերիաբանգում": "ամերիաբանկում",
        "ամերիաբանքում": "ամերիաբանկում",
        "ամերյա": "ամերիա",
        "ամերիաբանգի": "ամերիաբանկի",
        "ամերիաբանքի": "ամերիաբանկի",

        # ardshin
        "արդշինբանգ": "արդշինբանկ",
        "արդշինբանք": "արդշինբանկ",
        "արդշինբանգի": "արդշինբանկի",
        "արդշինբանքի": "արդշինբանկի",

        # idbank
        "այդի": "իդբանկ",
        "իդ": "իդբանկ",
        "իդի": "իդբանկ",
        "այդիբանկ": "իդբանկ",
        "իդբանգ": "իդբանկ",
        "իդբանք": "իդբանկ",
        "իդբանքի": "իդբանկի",
        "իդբանգի": "իդբանկի",

        # latin
        "ameria": "ամերիա",
        "ameriabank": "ամերիաբանկ",
        "ardshin": "արդշին",
        "ardshinbank": "արդշինբանկ",
    }

    normalized_words = []
    for word in words:
        normalized_words.append(token_fixes.get(word, word))

    text = " ".join(normalized_words)
    text = re.sub(r"\s+", " ", text).strip()
    return text
def num_to_armenian(n):
    if n == 0:
        return "զրո"

    ones = ["", "մեկ", "երկու", "երեք", "չորս", "հինգ", "վեց", "յոթ", "ութ", "ինը"]
    teens = ["տաս", "տասնմեկ", "տասներկու", "տասներեք", "տասնչորս", "տասնհինգ",
             "տասնվեց", "տասնյոթ", "տասնութ", "տասնինը"]
    tens = ["", "տաս", "քսան", "երեսուն", "քառասուն", "հիսուն",
            "վաթսուն", "յոթանասուն", "ութսուն", "իննսուն"]

    if n < 10:
        return ones[n]

    if n < 20:
        return teens[n - 10]

    if n < 100:
        return tens[n // 10] + (ones[n % 10] if n % 10 else "")

    if n < 1000:
        prefix = "" if n // 100 == 1 else ones[n // 100]
        rest = num_to_armenian(n % 100) if n % 100 else ""
        return (prefix + " հարյուր" + (" " + rest if rest else "")).strip()

    if n < 1000000:
        prefix = "" if n // 1000 == 1 else num_to_armenian(n // 1000)
        rest = num_to_armenian(n % 1000) if n % 1000 else ""
        return (prefix + " հազար" + (" " + rest if rest else "")).strip()

    return str(n)


def decimal_to_armenian(num_str):
    integer, fraction = num_str.split(".")
    int_part = num_to_armenian(int(integer))

    fraction = fraction.rstrip("0")
    if not fraction:
        return int_part

    frac_part = num_to_armenian(int(fraction))
    return f"{int_part} ամբողջ {frac_part}".replace("  ", " ")


def prepare_for_tts(text):
    # Clean formatting
    text = text.replace("**", "").replace("*", "")

    armenian_upper = "ԱԲԳԴԵԶԷԸԹԺԻԼԽԾԿՀՁՂՃՄՅՆՇՈՉՊՋՌՍՎՏՐՑՈՒՓՔՕՖ"
    armenian_lower = "աբգդեզէըթժիլխծկհձղճմյնշոչպջռսվտրցուփքօֆ"
    table = str.maketrans(armenian_upper, armenian_lower)
    text = text.translate(table)

    text = re.sub(r'\(([^)]+)\)', r', \1', text)
    text = re.sub(r'\[([^\]]+)\]', r'\1', text)
    text = re.sub(r'^\s*[-–]\s*', '', text, flags=re.MULTILINE)

    text = text.replace("մ/ճ", "մասնաճյուղ")
    text = text.replace("պող.", "պողոտա")
    text = text.replace("պող։", "պողոտա")
    text = text.replace("փ.", "փողոց")
    text = text.replace("փ։", "փողոց")
    text = text.replace("փող․", "փողոց")
    text = text.replace("ք.", "քաղաք")
    text = text.replace("ք։", "քաղաք")
    text = text.replace("հհ,", "հայաստան")
    text = text.replace("հհ.", "հայաստան")
    text = text.replace("հհ դրամ", "հայաստանի դրամ")
    text = text.replace("խճ․", "խճուղի")
    text = text.replace("խճ.", "խճուղի")
    text = text.replace("խճ։", "խճուղի")
    text = text.replace("շ.", "շենք")
    text = text.replace("շ։", "շենք")
    text = text.replace("գ.", "գյուղ")
    text = text.replace("գ։", "գյուղ")
    text = text.replace("հրպ.", "հրապարակ")
    text = text.replace("հրպ։", "հրապարակ")
    text = text.replace("մլն", "միլիոն")
    text = text.replace("մլրդ", "միլիարդ")
    text = text.replace("հզ.", "հազար")
    text = text.replace("թ.", "թվականին")

    text = text.replace("֏", " դրամ")
    text = text.replace("amd", "դրամ")
    text = text.replace("usd", "ամերիկյան դոլար")
    text = text.replace("$", "դոլար")
    text = text.replace("eur", "եվրո")
    text = text.replace("€", "եվրո")
    text = text.replace("rub", "ռուսական ռուբլի")
    text = text.replace("₽", "ռուբլի")
    text = text.replace("gbp", "բրիտանական ֆունտ")
    text = text.replace("£", "ֆունտ")
    text = text.replace("ամն դոլար", "ամերիկյան դոլար")

    # Remove thousand separators only for big numbers
    text = re.sub(r'(\d+)[.,](\d{3})[.,](\d{3})\b', r'\1\2\3', text)
    text = re.sub(r'(\d+)[.,](\d{3})\b', r'\1\2', text)

    def convert_time(m):
        h = num_to_armenian(int(m.group(1)))
        mins = int(m.group(2))
        if mins == 0:
            return h
        return f"{h} անց {num_to_armenian(mins)}"

    text = re.sub(r'\b(\d{1,2}):(\d{2})\b', convert_time, text)

    def convert_slash(m):
        a = num_to_armenian(int(m.group(1)))
        b = num_to_armenian(int(m.group(2)))
        return f"{a} սլեշ {b}"

    text = re.sub(r'\b(\d+)/(\d+)\b', convert_slash, text)

    def fmt_decimal(x: str) -> str:
        if "." in x:
            return decimal_to_armenian(x)
        return num_to_armenian(int(x))

    # Protect percentages/ranges with placeholders FIRST
    protected = {}

    def put(value: str) -> str:
        key = f"__PERCENTKEY{chr(65 + len(protected))}__"
        protected[key] = value
        return key

    # 9.37% - 9.43%
    text = re.sub(
        r'(\d+(?:\.\d+)?)\s*%\s*[-–]\s*(\d+(?:\.\d+)?)\s*%',
        lambda m: put(f"{fmt_decimal(m.group(1))}ից մինչև {fmt_decimal(m.group(2))} տոկոս"),
        text
    )

    # 9.37 տոկոս - 9.43 տոկոս
    text = re.sub(
        r'(\d+(?:\.\d+)?)\s*տոկոս\s*[-–]\s*(\d+(?:\.\d+)?)\s*տոկոս',
        lambda m: put(f"{fmt_decimal(m.group(1))}ից մինչև {fmt_decimal(m.group(2))} տոկոս"),
        text
    )

    # single %
    text = re.sub(
        r'(\d+(?:\.\d+)?)\s*%',
        lambda m: put(f"{fmt_decimal(m.group(1))} տոկոս"),
        text
    )

    # single written տոկոս
    text = re.sub(
        r'(\d+(?:\.\d+)?)\s*տոկոս',
        lambda m: put(f"{fmt_decimal(m.group(1))} տոկոս"),
        text
    )

    # Generic floats
    text = re.sub(
        r'\b\d+\.\d+\b',
        lambda m: decimal_to_armenian(m.group()),
        text
    )

    # Integer ranges
    text = re.sub(
        r'(\d+)\s*[-–]\s*(\d+)',
        lambda m: f"{num_to_armenian(int(m.group(1)))}ից մինչև {num_to_armenian(int(m.group(2)))}",
        text
    )

    def convert_int(m):
        n = int(m.group())
        if n >= 1000000:
            millions = n // 1000000
            remainder = n % 1000000
            result = num_to_armenian(millions) + " միլիոն"
            if remainder >= 1000:
                result += " " + num_to_armenian(remainder // 1000) + " հազար"
            elif remainder > 0:
                result += " " + num_to_armenian(remainder)
            return result
        if n >= 1000:
            thousands = n // 1000
            remainder = n % 1000
            result = num_to_armenian(thousands) + " հազար"
            if remainder > 0:
                result += " " + num_to_armenian(remainder)
            return result
        return num_to_armenian(n)

    text = re.sub(r'\b\d+\b', convert_int, text)

    # Restore protected percentages
    for key, value in protected.items():
        text = text.replace(key, value)

    # Ordinary colons only AFTER numbers are safe
    text = re.sub(r'\s*:\s*', '։ ', text)

    # Remove numbered list markers
    text = re.sub(r'\b(մեկ|երկու|երեք|չորս|հինգ|վեց|յոթ|ութ|ինը|տաս)\b\s*[.\-։]\s*', '', text)

    # Remove trailing helper phrases
    text = re.sub(
        r'կարող եմ նշել նաև այլ [^.։!?]*[.։!?]?$',
        '',
        text.strip(),
        flags=re.IGNORECASE
    )
    text = re.sub(
        r'(խորհուրդ եմ տալիս|այցելեք|դիմեք մասնաճյուղ)[^.։!?]*[.։!?]?$',
        '',
        text.strip(),
        flags=re.IGNORECASE
    )

    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\s+([,։])', r'\1', text)
    text = re.sub(r'([։])(?=\S)', r'\1 ', text)
    text = re.sub(r'։\s*։+', '։ ', text)
    text = re.sub(r'\s+,', ',', text)

    return text.strip()

def prepare_branches_for_tts(text):
    import re

    text = text.replace("**", "").replace("*", "")

    armenian_upper = "ԱԲԳԴԵԶԷԸԹԺԻԼԽԾԿՀՁՂՃՄՅՆՇՈՉՊՋՌՍՎՏՐՑՈՒՓՔՕՖ"
    armenian_lower = "աբգդեզէըթժիլխծկհձղճմյնշոչպջռսվտրցուփքօֆ"
    table = str.maketrans(armenian_upper, armenian_lower)
    text = text.translate(table)

    # Safe abbreviations
    text = text.replace("մ/ճ", "մասնաճյուղ")
    text = text.replace("պող.", "պողոտա")
    text = text.replace("փ.", "փողոց")
    text = text.replace("փող․", "փողոց")
    text = text.replace("ք.", "քաղաք")
    text = text.replace("խճ.", "խճուղի")
    text = text.replace("խճ․", "խճուղի")
    text = text.replace("շ.", "շենք")
    text = text.replace("տար.", "տարածք")
    text = text.replace("հեռ.", "հեռախոս")
    text = text.replace("հեռ․", "հեռախոս")
    text = text.replace("հհ, ", "")
    text = text.replace("հհ ", "")
    text = text.replace("🦽", "")
    text = text.replace("հրպ.", "հրապարակ")
    text = text.replace("հրպ․", "հրապարակ")
    text = text.replace("Ա. ", "Ա ")
    text = text.replace("Գ. ", "Գ ")

    # Time first
    def convert_time(m):
        h = num_to_armenian(int(m.group(1)))
        mins = int(m.group(2))
        if mins == 0:
            return h
        return f"{h} անց {num_to_armenian(mins)}"

    text = re.sub(r'\b(\d{1,2}):(\d{2})\b', convert_time, text)

    # Ordinary colons -> pause
    text = re.sub(r'\s*:\s*', '։ ', text)

    # Address slash: 56/162 -> հիսունվեց սլեշ հարյուր վաթսուներկու
    text = re.sub(
        r'\b(\d+)/(\d+)\b',
        lambda m: f"{num_to_armenian(int(m.group(1)))} սլեշ {num_to_armenian(int(m.group(2)))}",
        text
    )

    # Ordinals
    text = re.sub(r'\b1-ին\b', 'առաջին', text)
    text = re.sub(r'\b2-րդ\b', 'երկրորդ', text)
    text = re.sub(r'\b3-րդ\b', 'երրորդ', text)
    text = re.sub(r'\b4-րդ\b', 'չորրորդ', text)

    # N 3 -> էն երեք
    text = re.sub(
        r'\bN\s*(\d+)\b',
        lambda m: f"էն {num_to_armenian(int(m.group(1)))}",
        text
    )

    # Standalone integers
    text = re.sub(
        r'\b\d+\b',
        lambda m: num_to_armenian(int(m.group())),
        text
    )

    # Better branch pauses
    text = text.replace(" մասնաճյուղ — ", " մասնաճյուղ։ ")
    text = text.replace(" մասնաճյուղ՝ ", " մասնաճյուղ։ ")
    text = text.replace("գլխամասային գրասենյակ — ", "գլխամասային գրասենյակ։ ")
    text = text.replace("գլխամասային գրասենյակ՝ ", "գլխամասային գրասենյակ։ ")

    # Each address-like line should become a separate sentence
    text = re.sub(r'\n+', '։ ', text)

    # Clean spacing
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'։\s*։+', '։ ', text)
    text = re.sub(r'\s+([,։])', r'\1', text)
    text = re.sub(r'([։])(?=\S)', r'\1 ', text)

    return text.strip()


def search(query, n_results=8):
    query = normalize_text(query)
    query_lower = query.lower()
    words = query_lower.split()

    all_docs = collection.get(include=["documents"])["documents"]

    bank_keywords = {
        "ամերիաբանկ": "Ամերիաբանկ",
        "ամերիաբանք": "Ամերիաբանկ",
        "ամերիաբանգ": "Ամերիաբանկ",
        "ամերիաբանկը": "Ամերիաբանկ",
        "ամերիաբանկի": "Ամերիաբանկ",
        "ամերիա": "Ամերիաբանկ",
        "ameria": "Ամերիաբանկ",

        "արդշինբանկ": "Արդշինբանկ",
        "արդշինբանկը": "Արդշինբանկ",
        "արդշինբանկի": "Արդշինբանկ",
        "արդշին բանկ": "Արդշինբանկ",
        "արդշինբանք": "Արդշինբանկ",
        "արդշին բանք": "Արդշինբանկ",
        "արդշինբանգ": "Արդշինբանկ",
        "արդշին բանգ": "Արդշինբանկ",
        "արդշին": "Արդշինբանկ",
        "ardshin": "Արդշինբանկ",

        "այդի բանկ": "ԱյԴի Բանկ",
        "այդի բանկը": "ԱյԴի Բանկ",
        "այդի բանկի": "ԱյԴի Բանկ",
        "այդի բանք": "ԱյԴի Բանկ",
        "այդի բանգ": "ԱյԴի Բանկ",
        "այդի": "ԱյԴի Բանկ",
        "իդ բանկ": "ԱյԴի Բանկ",
        "իդբանկ": "ԱյԴի Բանկ",
        "իդբանկի": "ԱյԴի Բանկ",
        "idbank": "ԱյԴի Բանկ",
        "id bank": "ԱյԴի Բանկ",
        "էյդի": "ԱյԴի Բանկ",
    }

    topic_keywords = {
        "մասնաճյուղ": "մասնաճյուղեր",
        "մասնաճյուղը": "մասնաճյուղեր",
        "մասնաճյուղի": "մասնաճյուղեր",
        "մասնաճյուղերը": "մասնաճյուղեր",
        "մասնաճյուղեր": "մասնաճյուղեր",
        "մասնաճու": "մասնաճյուղեր",
        "մասնաչու": "մասնաճյուղեր",
        "մասնաճ": "մասնաճյուղեր",
        "ֆիլիալ": "մասնաճյուղեր",
        "հասցե": "մասնաճյուղեր",
        "հասցեն": "մասնաճյուղեր",
        "հասցեները": "մասնաճյուղեր",
        "աշխատանքային ժամ": "մասնաճյուղեր",
        "աշխատաժամ": "մասնաճյուղեր",

        "ավանդ": "ավանդներ",
        "ավանդը": "ավանդներ",
        "ավանդի": "ավանդներ",
        "ավանդներ": "ավանդներ",
        "ավանդների": "ավանդներ",
        "ավանդային": "ավանդներ",
        "խնայողական": "ավանդներ",
        "ցպահանջ": "ավանդներ",
        "ներդնել": "ավանդներ",

        "վարկ": "վարկեր",
        "վարկը": "վարկեր",
        "վարկի": "վարկեր",
        "վարկեր": "վարկեր",
        "վարկերի": "վարկեր",
        "վարգ": "վարկեր",
        "կրեդիտ": "վարկեր",
        "loan": "վարկեր",
        "ուսանողական": "վարկեր",
        "սպառողական": "վարկեր",
        "վարկային": "վարկեր",
    }

    bank_aliases_for_filter = {
        "Ամերիաբանկ": ["բանկ: ամերիաբանկ", "ամերիաբանկ", "ամերիա"],
        "Արդշինբանկ": ["բանկ: արդշինբանկ", "արդշինբանկ", "արդշին"],
        "ԱյԴի Բանկ": ["բանկ: այդի բանկ", "այդի բանկ", "իդբանկ", "idbank", "id bank"],
    }

    topic_aliases_for_filter = {
        "մասնաճյուղեր": ["թեմա: մասնաճյուղեր", "մասնաճյուղ", "հասցե", "ֆիլիալ"],
        "ավանդներ": ["թեմա: ավանդներ", "ավանդ", "խնայողական", "ցպահանջ"],
        "վարկեր": ["թեմա: վարկեր", "վարկ", "ուսանողական", "սպառողական", "կրեդիտ", "վարկային"],
    }

    def token_matches(keyword, words):
        if keyword in words:
            return True
        for w in words:
            if w.startswith(keyword):
                return True
        return False

    is_comparison = re.search(
        r'համեմատ|ավելի լավ|ավելի բարձր|ավելի ցածր|որ բանկ|ո՞ր բանկ|համեմատիր|տարբերությունը|թե',
        query_lower
    ) is not None

    mentioned_banks = []
    for keyword, bank_name in bank_keywords.items():
        if " " in keyword:
            if keyword in query_lower and bank_name not in mentioned_banks:
                mentioned_banks.append(bank_name)
        else:
            if token_matches(keyword, words) and bank_name not in mentioned_banks:
                mentioned_banks.append(bank_name)

    topic_scores = {
        "մասնաճյուղեր": 0,
        "ավանդներ": 0,
        "վարկեր": 0,
    }

    for keyword, topic_name in topic_keywords.items():
        if " " in keyword:
            if keyword in query_lower:
                topic_scores[topic_name] += 2
        else:
            if token_matches(keyword, words):
                topic_scores[topic_name] += 1

    topic_filter = None
    best_topic = max(topic_scores, key=topic_scores.get)
    if topic_scores[best_topic] > 0:
        topic_filter = best_topic

    query_embedding = model.encode(query).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(max(n_results * 3, n_results), 24)
    )
    chroma_docs = results["documents"][0]

    filtered = []

    for doc in all_docs:
        doc_lower = doc.lower()

        if is_comparison and mentioned_banks:
            bank_match = False
            for bank in mentioned_banks:
                if any(alias in doc_lower for alias in bank_aliases_for_filter.get(bank, [])):
                    bank_match = True
                    break
        else:
            if mentioned_banks:
                bank_match = any(alias in doc_lower for alias in bank_aliases_for_filter.get(mentioned_banks[0], []))
            else:
                bank_match = True

        if topic_filter is None:
            topic_match = True
        else:
            topic_match = any(alias in doc_lower for alias in topic_aliases_for_filter.get(topic_filter, []))

        if bank_match and topic_match:
            filtered.append(doc)

    if filtered:
        reranked = [doc for doc in chroma_docs if doc in filtered]
        if reranked:
            return reranked[:n_results]
        return filtered[:n_results]

    return chroma_docs[:n_results]



def ask(question):
    print("ASK FUNCTION CALLED")
    question_normalized = normalize_text(question)

    deposit_open_patterns = [
        "որտեղ կարելի է ավանդ բացել",
        "որտեղ բացել ավանդ",
        "ինչպես բացել ավանդ",
        "ավանդ բացել",
        "որտեղ կարելի է ավանդը բացել",
        "որտեղ բացել ավանդը",
        "ինչպես բացել ավանդը",
        "ավանդը բացել",
    ]

    if any(p in question_normalized for p in deposit_open_patterns):
        if "ավանդ" in question_normalized:
            question_normalized += " ավանդների պայմանները և բացման հնարավորությունը"

    is_deposit_max_question = (
        "ավանդ" in question_normalized and
        any(x in question_normalized for x in ["մինչև", "առավելագույն", "ամենաշատ", "մաքսիմում"])
    )

    is_deposit_list_question = (
        "ավանդ" in question_normalized and
        any(x in question_normalized for x in [
            "ինչ ավանդներ կան",
            "ավանդների տեսակները",
            "ինչ տեսակ",
            "տեսակները",
            "նշիր ավանդները",
            "նշիր ավանդների տեսակները",
            "ավանդ",
        ])
    )

    if is_deposit_list_question and any(x in question_normalized for x in [
        "տոկոս", "տոկոսադրույք", "առավելագույն", "մաքսիմում", "մինչև", "գումար"
    ]):
        is_deposit_list_question = False

    comparison_keywords = [
        "համեմատ", "ավելի լավ", "ավելի բարձր", "ավելի ցածր",
        "որ բանկ", "ո՞ր բանկ", "ամենաբարձր", "ամենացածր",
        "համեմատիր", "տարբերությունը", "ինչով է տարբերվում",
        "թե", "ավելի շահավետ", "որն է լավ"
    ]

    calculation_keywords = [
        "հաշվիր", "հաշվե", "եկամուտ", "կստանամ", "կստանա",
        "շահույթ", "կվճարեմ", "կվճարի", "որքան կստան",
        "եթե ներդնեմ", "եթե դնեմ", "եթե ավանդ դնեմ",
        "մեկ տարում", "քանի դրամ", "որքան գումար",
        "որքա՞ն եկամուտ", "ինչքան եկամուտ", "մեկ միլիոն",
        "ներդնեմ", "եկամութ", "եկամուտ կունենամ"
    ]

    is_comparison = any(kw in question_normalized for kw in comparison_keywords)
    is_calculation = any(kw in question_normalized for kw in calculation_keywords)
    is_rate_question = "տոկոս" in question_normalized or "տոկոսադրույք" in question_normalized

    is_broad_deposit_rate_question = (
        "ավանդ" in question_normalized and
        is_rate_question and
        not any(x in question_normalized for x in [
            "դրամ", "դոլար", "եվրո", "ռուբլ",
            "ամսական", "վերջում", "ժամկետ",
            "ապահով", "կուտակային", "խնայողական"
        ])
    )

    is_broad_loan_rate_question = (
        "վարկ" in question_normalized and
        is_rate_question and
        not any(x in question_normalized for x in [
            "ուսանողական", "սպառողական", "անգրավ", "գրավ", "վարկային գիծ"
        ])
    )

    if is_rate_question and not any(x in question_normalized for x in ["ավանդ", "վարկ"]):
        return "Խնդրում եմ նշեք՝ ավանդի՞, թե՞ վարկի տոկոսադրույքն է ձեզ հետաքրքրում։"

    if is_broad_deposit_rate_question:
        return "Խնդրում եմ նշեք՝ ո՞ր ավանդի, արժույթի կամ ժամկետի տոկոսադրույքն է ձեզ հետաքրքրում։"

    if is_broad_loan_rate_question:
        return "Խնդրում եմ նշեք՝ ո՞ր վարկատեսակի տոկոսադրույքն է ձեզ հետաքրքրում։"

    logical_instructions = ""

    if is_comparison:
        relevant_chunks = search(question_normalized, n_results=12)
    elif is_calculation:
        relevant_chunks = search(question_normalized, n_results=10)
    elif is_deposit_list_question:
        relevant_chunks = search(question_normalized, n_results=10)
    else:
        relevant_chunks = search(question_normalized, n_results=5)

    if not relevant_chunks:
        return "Տվյալ տեղեկատվությունը հասանելի չէ"

    context = "\n\n---\n\n".join(relevant_chunks)

    if is_deposit_max_question:
        if not any(x in context.lower() for x in ["առավելագույն", "մաքսիմում", "մինչև"]):
            return "Տվյալ տեղեկատվությունը հասանելի չէ։"

    if is_calculation:
        logical_instructions = (
            "\nՀԱՇՎԱՐԿԻ ԿԱՆՈՆՆԵՐ:\n"
            "- Կատարիր հաշվարկ միայն տրամադրված տվյալներով։\n"
            "- Եթե հարցում կա մի քանի հնարավոր տոկոսադրույք, մի հորինիր մեկը։ Հստակ ասա, որ պետք է նշել ավանդի տեսակը, արժույթը կամ ժամկետը։\n"
            "- Եթե կա մեկ հստակ տոկոսադրույք, հաշվարկիր այն։\n"
            "- Եթե կա տոկոսադրույքի միջակայք, կարող ես տալ նվազագույնից մինչև առավելագույն արդյունքը։\n"
            "- Մի հորինիր բացակայող թվեր։\n"
            "- Տուր կարճ և հստակ պատասխան։\n"
        )
    elif is_comparison:
        logical_instructions = (
            "\nՀԱՄԵՄԱՏՈՒԹՅԱՆ ԿԱՆՈՆՆԵՐ:\n"
            "- Եթե հարցում նշված են երկու կամ ավելի բանկեր, պարտադիր համեմատի՛ր բոլոր նշված բանկերը։\n"
            "- Մի սահմանափակվիր միայն մեկ բանկով։\n"
            "- Մի ասա՝ այցելեք կայքը, դիմեք մասնաճյուղ, կամ այլ ընդհանուր խորհուրդներ։\n"
            "- Օգտագործիր միայն context-ում առկա տվյալները։\n"
            "- Եթե մեկ բանկի տվյալը բացակայում է, հստակ ասա՝ տվյալը նշված չէ, բայց շարունակիր համեմատել մյուս բանկերի առկա տվյալները։\n"
            "- Եթե հարցը «որն է ավելի լավ» ձևով է, ասա՝ որ դեպքում որն է ավելի շահավետ՝ ըստ տոկոսադրույքի, առավելագույն գումարի, ժամկետի կամ հատուկ պայմանների։\n"
            "- Պատասխանը կառուցիր կարճ համեմատությամբ։\n"
        )
    elif is_deposit_list_question:
        logical_instructions = (
            "\nԱՎԱՆԴՆԵՐԻ ԿԱՆՈՆՆԵՐ:\n"
            "- Նշիր բոլոր առկա տեսակները։\n"
            "- Մի սահմանափակվիր միայն մեկ օրինակով։\n"
            "- Մի համարակալիր «մեկ, երկու, երեք» բառերով։\n"
            "- Մի ավելացրու վերջում հավելյալ բացատրություն։\n"
        )
    elif is_rate_question:
        logical_instructions = (
            "\nՏՈԿՈՍԱԴՐՈՒՅՔԻ ԿԱՆՈՆՆԵՐ:\n"
            "- Մի կարդա ամբողջ աղյուսակը։\n"
            "- Տուր համառոտ պատասխան՝ հիմնական միջակայքով կամ առավելագույն արժեքով։\n"
            "- Մի համարակալիր կետերը «մեկ, երկու, երեք» բառերով։\n"
            "- Մի ավելացրու վերջում խորհուրդ կամ ավելորդ բացատրություն։\n"
        )

    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": (
                    "Դու հայկական բանկային օգնական ես։\n"
                    "Կարող ես պատասխանել միայն վարկերի, ավանդների և մասնաճյուղերի վերաբերյալ հարցերին։\n"
                    "Պատասխանիր միայն հայերեն։\n"
                    "Օգտագործիր միայն ստորև տրված տեղեկատվությունը։\n"
                    "Խստիվ արգելվում է որևէ բան հորինել կամ օգտագործել արտաքին գիտելիք։\n"
                    "Եթե հարցը դուրս է թեմայից, ասա՝ «Կարող եմ օգնել միայն վարկերի, ավանդների և մասնաճյուղերի վերաբերյալ հարցերով»։\n"
                    "Եթե պատասխանը տվյալների մեջ չկա, ասա՝ «Տվյալ տեղեկատվությունը հասանելի չէ»։\n"
                    "Մի օգտագործիր համարակալում՝ «մեկ, երկու, երեք»։\n"
                    "Մի ավարտիր պատասխանն ավելորդ մեկնաբանությամբ, ներողությամբ կամ խորհրդով։\n"
                    + logical_instructions +
                    f"\nՏեղեկատվություն:\n{context}"
                )
            },
            {"role": "user", "content": question}
        ],
        temperature=0.1
    )

    answer = response.choices[0].message.content.strip()

    print("RAW ANSWER BEFORE TTS:", repr(answer))

    with open("debug_raw_answer.txt", "a", encoding="utf-8") as f:
        f.write("RAW ANSWER: " + repr(answer) + "\n")

    # then cleanup + TTS

    fallback_phrases = [
        "կարող եմ օգնել միայն",
        "տվյալ տեղեկատվությունը հասանելի չէ",
        "չեմ կարող պատասխանել",
        "այցելեք",
        "դիմեք մասնաճյուղ",
        "պաշտոնական կայքը",
        "խորհուրդ եմ տալիս",
    ]

    sentences = [s.strip() for s in re.split(r'[։.!?]+', answer) if s.strip()]
    cleaned = []
    for s in sentences:
        low = s.lower()
        if any(p in low for p in fallback_phrases):
            if cleaned:
                break
        cleaned.append(s)

    if cleaned:
        answer = "։ ".join(cleaned).strip()
        if not answer.endswith("։"):
            answer += "։"

    if "մասնաճյուղ" in question_normalized or "հասցե" in question_normalized:
        final_answer = prepare_branches_for_tts(answer)
        print("FINAL ANSWER AFTER TTS:", repr(final_answer))
        return final_answer

    final_answer = prepare_for_tts(answer)
    print("FINAL ANSWER AFTER TTS:", repr(final_answer))
    return final_answer
load_data()