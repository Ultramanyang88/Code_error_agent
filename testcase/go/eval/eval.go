package eval

import (
	"fmt"
	"strconv"
	"strings"
	"unicode"
)

// Eval evaluates a simple math expression and returns the result.
// Supports: +, -, *, / operators and integer numbers.
func Eval(expr string) (float64, error) {
	expr = strings.ReplaceAll(expr, " ", "")
	if expr == "" {
		return 0, fmt.Errorf("empty expression")
	}

	tokens, err := tokenize(expr)
	if err != nil {
		return 0, err
	}

	return evaluate(tokens)
}

type token struct {
	isNumber bool
	number   float64
	operator rune
}

func tokenize(expr string) ([]token, error) {
	var tokens []token
	i := 0

	for i < len(expr) {
		ch := rune(expr[i])

		if unicode.IsDigit(ch) || ch == '.' {
			// Parse number
			j := i
			for j < len(expr) && (unicode.IsDigit(rune(expr[j])) || expr[j] == '.') {
				j++
			}
			numStr := expr[i:j]
			num, err := strconv.ParseFloat(numStr, 64)
			if err != nil {
				return nil, fmt.Errorf("invalid number: %s", numStr)
			}
			tokens = append(tokens, token{isNumber: true, number: num})
			i = j
		} else if ch == '+' || ch == '-' || ch == '*' || ch == '/' {
			tokens = append(tokens, token{isNumber: false, operator: ch})
			i++
		} else {
			return nil, fmt.Errorf("invalid character: %c", ch)
		}
	}

	return tokens, nil
}

func evaluate(tokens []token) (float64, error) {
	if len(tokens) == 0 {
		return 0, fmt.Errorf("no tokens to evaluate")
	}

	if !tokens[0].isNumber {
		return 0, fmt.Errorf("expression must start with a number")
	}

	result := tokens[0].number

	for i := 1; i < len(tokens); i += 2 {
		if i >= len(tokens) {
			break
		}

		op := tokens[i]
		if op.isNumber {
			return 0, fmt.Errorf("expected operator at position %d", i)
		}

		if i+1 >= len(tokens) {
			return 0, fmt.Errorf("missing number after operator")
		}

		num := tokens[i+1]
		if !num.isNumber {
			return 0, fmt.Errorf("expected number at position %d", i+1)
		}

		switch op.operator {
		case '+':
			result += num.number
		case '-':
			result -= num.number
		case '*':
			result *= num.number
		case '/':
			if num.number == 0 {
				return 0, fmt.Errorf("division by zero")
			}
			result /= num.number
		}
	}

	return result, nil
}
