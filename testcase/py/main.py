#!/usr/bin/env python3
import sys
from eval_module import eval_expr, EvalError


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <math_expression>", file=sys.stderr)
        print(f'Example: {sys.argv[0]} "2+3"', file=sys.stderr)
        sys.exit(1)

    expr = sys.argv[1]
    try:
        result = eval_expr(expr)
    except EvalError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"{expr} = {result}")


if __name__ == "__main__":
    main()

