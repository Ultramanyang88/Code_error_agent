package main

import (
	"fmt"
	"os"

	"demo/eval"
)

func main() {
	if len(os.Args) != 2 {
		fmt.Fprintf(os.Stderr, "Usage: %s <math_expression>\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "Example: %s \"2+3*4\"\n", os.Args[0])
		os.Exit(1)
	}

	expr := os.Args[1]
	result, err := eval.Eval(expr)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("%s = %v\n", expr, result)
}
