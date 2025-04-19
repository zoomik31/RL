maps = {"souless\map_1.xlsx": [(420, 395), 0], 
        "souless\map_2.xlsx": [(420, 455), 1],
        "souless\map_4.xlsx": [(420, 515), 2]}

x = 0
for number, (key, value) in enumerate(maps.items()):
    print(value[0][0])