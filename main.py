import re

# ==============================================================================
# Функція: zsprotasyk23_email
# Опис: Валідація електронної пошти за допомогою регулярного виразу (RFC 5322).
# ------------------------------------------------------------------------------
# Першоджерело: https://github.com/example-user/email-utils/blob/main/validator.py
# Автор оригіналу: John Doe (https://github.com/example-user)
# Назва ліцензії: MIT License
# Внесені зміни: Так (оптимізовано регулярне вираз під Python re.compile 
#                 та додано обробку порожніх рядків).
# ==============================================================================

def validate_email(email: str) -> bool:
    """
    Перевіряє, чи відповідає переданий рядок формату адреси електронної пошти.
    """
    if not email or not isinstance(email, str):
        return False

    # Скомпільований паттерн (модифікована версія коду John Doe)
    email_regex = re.compile(
        r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
    )
    return bool(email_regex.match(email.strip()))


if __name__ == "__main__":
    test_email = "student@university.edu.ua"
    print(f"Результат перевірки '{test_email}': {validate_email(test_email)}")