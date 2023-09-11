from colorama import Back, Fore, Style


def get_input(prefix: str = "") -> str:
    print(Fore.RED, flush=True, end="")
    content = input(prefix)
    print(Style.RESET_ALL, flush=True, end="")
    return content


def print_red(msg: str, **args):
    print(Fore.RED + msg + Style.RESET_ALL, flush=True, **args)


def print_error(msg: str, **args):
    print(Fore.RED + msg + Style.RESET_ALL, flush=True, **args)


def print_warning(msg: str, **args):
    print(Fore.YELLOW + msg + Style.RESET_ALL, flush=True, **args)


def print_info(msg: str, **args):
    print(Fore.YELLOW + msg + Style.RESET_ALL, flush=True, **args)


def print_system(msg: str, **args):
    print(Back.LIGHTBLACK_EX + msg + Style.RESET_ALL, flush=True, **args)


def print_human(msg: str, **args):
    print(Fore.GREEN + msg + Style.RESET_ALL, flush=True, **args)


def print_ai(msg: str, **args):
    print(Fore.BLUE + msg + Style.RESET_ALL, flush=True, **args)
