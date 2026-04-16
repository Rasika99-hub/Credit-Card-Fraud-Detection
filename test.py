# test.py — paste this and run: python test.py
import os
import stat

path = r"C:\Credit Card Fraud Detection\creditcard.csv"

print("File exists    :", os.path.exists(path))
print("File size      :", os.path.getsize(path), "bytes")
print("Is readable    :", os.access(path, os.R_OK))
print("Is writable    :", os.access(path, os.W_OK))
print("File stats     :", oct(os.stat(path).st_mode))