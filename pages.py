
start = 471
end = 547
page_gap = 2

count = 0
for i in range(start, end, page_gap*2):
    print(str(i) + "-" + str(i+page_gap-1), end=", ")
    count += 1

print("\n\n", count)
print("\n")
count = 0
for i in range(start, end, page_gap*2):
    print(str(i+page_gap) + "-" + str(i+page_gap+page_gap-1), end=", ")
    count += 1

print("\n\n", count)
