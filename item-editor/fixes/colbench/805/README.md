# Task 805 Fix: Shadow Calculation with Hidden Formula

## IFE Summary

The hidden reference solution contains a specific formula that the simulated user not only failed to communicate but actively contradicted:

```python
def calculate_shadow(border_thickness, title_bar_height, window_height):
    return (title_bar_height + 2 * border_thickness, max(10, window_height // 4))
```

The simulated user:
1. Said "I don't know about any specific conventions or preferred ratios for the shadow extension"
2. Endorsed a constant return: "I think using the default 20 points for both horizontal and vertical shadow extensions would be fine"

## Evidence

- Hidden formula: `(title_bar_height + 2*border_thickness, max(10, window_height // 4))`
- User response: "I don't have a specific formula, but I think a reasonable heuristic would be fine"
- User endorsed: `return (20, 20)` - a constant that completely ignores input parameters

## Fix Strategy

The formula is not derivable from UI/UX principles without specification. The user actively misdirected the agent toward a constant return value.

## Fix Applied

- `instruction_override.json`: Specifies the exact formula for shadow dimensions
