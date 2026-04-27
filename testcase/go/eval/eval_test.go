package eval

import (
	"testing"
)

func TestEvalSimple(t *testing.T) {
	tests := []struct {
		name     string
		expr     string
		expected float64
	}{
		{"addition", "2+3", 5},
		{"subtraction", "10-4", 6},
		{"multiplication", "3*4", 12},
		{"division", "20/5", 4},
		{"precedence", "2+3*4", 14},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := Eval(tt.expr)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if result != tt.expected {
				t.Errorf("Eval(%q) = %v, want %v", tt.expr, result, tt.expected)
			}
		})
	}
}
