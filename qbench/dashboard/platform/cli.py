"""Private, interactive first-administrator bootstrap. No passwords in argv."""
import argparse
import getpass
import sys

from .store import PlatformError, Store


def main(argv=None):
    parser = argparse.ArgumentParser(prog="qbench-admin")
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("init", "create a new private database and first administrator"),
        ("recover-admin", "host-only recovery of an enabled administrator; revokes sessions"),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--database", required=True)
        command.add_argument("--username", required=True)
    args = parser.parse_args(argv)
    try:
        password = getpass.getpass("Administrator password (12–128 characters): ")
        confirm = getpass.getpass("Confirm password: ")
        if password != confirm:
            raise PlatformError("Passwords do not match.")
        if args.command == "init":
            Store.initialize(args.database, args.username, password)
        else:
            Store(args.database).recover_admin(args.username, password)
    except (PlatformError, OSError, EOFError) as exc:
        print(f"qbench-admin: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 1
    print("Private platform database initialized. No default credentials were created." if args.command == "init"
          else "Administrator recovery complete. Sessions revoked; change the temporary password at sign-in.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
