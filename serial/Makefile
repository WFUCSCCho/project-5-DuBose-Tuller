CC      = gcc
CFLAGS  = -O0 -Wall -std=c99
TARGET  = batched_ttv

$(TARGET): batched_ttv.c
	$(CC) $(CFLAGS) -o $(TARGET) batched_ttv.c

clean:
	rm -f $(TARGET) results.csv
