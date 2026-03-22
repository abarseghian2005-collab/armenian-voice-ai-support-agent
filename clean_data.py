import os
import re

# Lines to remove (navigation menu garbage)
remove_patterns = [
    "ԱՎԱՆԴՆԵՐ", "ԲԻԶՆԵՍ", "ՆԵՐԴՐՈՒՄ", "ՀԱՅ", "IR",
    "Ձեր համար", "Ձեր բիզնեսի", "Ստատուս | Plus",
    "ԱՎԱՆԴՆԵՐ", "ՆԵՐԴՐՈՒՄՆԵՐ", "ՔԱՐՏ", "ՎԱՐՔԵՐ", "ԱՅԼ",
    "I-BANKING", "ԴԱՐՆԱԼ ՀԱՉԱՀՈՐԴ", "ՍՏԱՆԱԼ ՎԱՐՔ",
    "ԱՆՀԱՏԱՆԵՐԻՆ", "ԿԱՐ", "Հայերեն",
    "expand_more", "🦽",
    "ՍՈՑԻԱԼ. ՑԱՆՑԵՐ",
    "Ուշադրություն",
]

remove_exact = [
    "ԱՎԱՆԴՆԵՐ", "ԲԻԶՆԵՍ", "ՆԵՐԴՐՈՒՄ", "ՀԱՅ", "IR",
    "Ձեր համար", "Ձեր բիզնեսի համար", "Ստատուս | Plus",
    "ԱՎԱՆԴՆԵՐ", "ՆԵՐԴՐՈՒՄՆԵՐ", "ՔԱՐՏ", "ՎԱՐՔԵՐ", "ԱՅԼ",
    "I-BANKING", "ԴԱՐՆԱԼ ՀԱՉԱՀՈՐԴ", "ՍՏԱՆԱԼ ՎԱՐՔ",
    "ԱՆԳՐԱՎ", "ԿԱՀԻԿՈՎ ԱՊԱՀՈՎԱԴ",
    "Հայերեն", "expand_more",
]

data_folder = "./data"
for filename in os.listdir(data_folder):
    if filename.endswith(".txt"):
        filepath = os.path.join(data_folder, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()

        cleaned = []
        for line in lines:
            stripped = line.strip()
            # Skip short navigation lines (less than 4 chars)
            if len(stripped) < 4:
                continue
            # Skip lines that are ALL CAPS and short (navigation items)
            if stripped.isupper() and len(stripped) < 30:
                continue
            # Skip lines with just symbols
            if stripped in ["©", "🦽", "expand_more", "►"]:
                continue
            cleaned.append(line)

        with open(filepath, "w", encoding="utf-8") as f:
            f.writelines(cleaned)

        print(f"Cleaned {filename}: {len(lines)} -> {len(cleaned)} lines")

print("Done!")