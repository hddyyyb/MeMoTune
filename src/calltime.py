def call_counter(func):
    def wrapper(*args, **kwargs):
        wrapper.calls += 1
        return func(*args, **kwargs)
    wrapper.calls = 0
    return wrapper

@call_counter
def my_function():
    print("Function is called")

# Calling the function multiple times
my_function()
my_function()
my_function()

print(f"The function was called {my_function.calls} times.")
