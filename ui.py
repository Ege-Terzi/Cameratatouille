# Top margin for overlay text placement
TOP_MARGIN = 40

# Left margin for overlay text placement
LEFT_MARGIN = 25

# Vertical spacing between information lines
INFO_LINE_HEIGHT = 28

# Vertical spacing between recipe step lines
STEP_LINE_HEIGHT = 28

# Bottom margin to leave space for help text
BOTTOM_MARGIN = 60

# Offset from the bottom for help text placement
HELP_TEXT_Y_OFFSET = 20

# Overlay rectangle top boundary
OVERLAY_TOP = 10

# Overlay rectangle left boundary
OVERLAY_LEFT = 10

# Overlay rectangle right boundary
OVERLAY_RIGHT = 760

# Overlay rectangle bottom boundary
OVERLAY_BOTTOM = 420


def choose_recipe_from_terminal(recipes):
    # Display all available recipes in the terminal
    print("\nAvailable recipes:\n")

    # Loop through recipes and print their details
    for i, recipe in enumerate(recipes, start=1):
        required = ", ".join(recipe["ingredients"])
        optional = ", ".join(recipe.get("optional_ingredients", []))

        print(f"{i}. {recipe['name']}")
        print(f"   Required ingredients: {required}")

        # Print optional ingredients only if they exist
        if optional:
            print(f"   Optional ingredients: {optional}")
        print()

    # Ask the user to select a recipe until a valid input is entered
    while True:
        try:
            choice = int(input("Select a recipe by number: "))

            # Return zero-based recipe index if valid
            if 1 <= choice <= len(recipes):
                return choice - 1

            print("Please enter a valid recipe number.")
        except ValueError:
            # Handle non-integer input
            print("Please enter a number.")


def build_info_lines(selected_recipe, stable_ingredients, stable_bowl_state, missing, current_step_idx, auto_mode):
    # Build the list of information lines shown on the video overlay
    return [
        f"Selected recipe: {selected_recipe['name']}",
        f"Detected ingredients: {', '.join(stable_ingredients) if stable_ingredients else '-'}",
        f"Bowl state: {stable_bowl_state}",
        f"Required: {', '.join(selected_recipe['ingredients'])}",
        f"Missing: {', '.join(missing) if missing else 'none'}",
        f"Current step: {current_step_idx + 1}/{len(selected_recipe['steps'])}",
        f"Auto mode: {'ON' if auto_mode else 'OFF'}",
    ]


def build_step_line(step_index, step, completed_steps, current_step_idx):
    # Mark the step as completed or not completed
    prefix = "[x]" if step_index in completed_steps else "[ ]"

    # Highlight the current active step with an arrow
    marker = "->" if step_index == current_step_idx else "  "

    # Return formatted step line
    return f"{marker} {prefix} {step_index + 1}. {step['text']}"


def get_help_text():
    # Return keyboard shortcut help text shown at the bottom of the screen
    return "[n] next  [p] previous  [c] toggle complete  [a] auto  [space] pause  [q] quit"


def get_step_color(step_index, completed_steps, current_step_idx):
    # Return green for completed steps
    if step_index in completed_steps:
        return (0, 255, 0)

    # Return yellow for the current active step
    if step_index == current_step_idx:
        return (0, 255, 255)

    # Return white for all other steps
    return (255, 255, 255)


def get_step_start_y(info_lines_count):
    # Compute the vertical starting position for recipe step text
    return TOP_MARGIN + info_lines_count * INFO_LINE_HEIGHT + 28


def get_max_visible_steps(frame_height, step_start_y):
    # Compute how much vertical space remains for displaying recipe steps
    available_height = frame_height - step_start_y - BOTTOM_MARGIN

    # Return the maximum number of visible steps, at least 1
    return max(1, available_height // STEP_LINE_HEIGHT)


def get_visible_step_range(total_steps, current_step_idx, max_visible_steps):
    # Limit the context window to at most 3 visible steps
    context_window = min(max_visible_steps, 3)

    # Start by centering the visible range around the current step
    start_idx = max(0, current_step_idx - 1)
    end_idx = min(total_steps, current_step_idx + 2)

    # Expand upward if there is still room in the visible window
    while end_idx - start_idx < context_window and start_idx > 0:
        start_idx -= 1

    # Expand downward if there is still room in the visible window
    while end_idx - start_idx < context_window and end_idx < total_steps:
        end_idx += 1

    # Return the start and end indices of visible steps
    return start_idx, end_idx