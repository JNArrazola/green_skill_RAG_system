def modify_set(my_set):
    my_set.add(4)
    print(f"Inside function (modified): {my_set}")

original_set = {1, 2, 3}
print(f"Original set (before call): {original_set}")
modify_set(original_set)
print(f"Original set (after call): {original_set}")