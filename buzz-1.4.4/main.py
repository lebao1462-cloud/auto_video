import sys

if len(sys.argv) > 1 and sys.argv[1] == "--argos-worker":
    from buzz.argos_worker import main as argos_worker_main

    raise SystemExit(argos_worker_main(sys.argv[2:]))

import buzz.buzz

if __name__ == "__main__":
    buzz.buzz.main()
