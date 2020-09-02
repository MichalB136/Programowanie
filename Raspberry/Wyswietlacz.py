from RPLCD.gpio import CharLCD
from RPi import GPIO
import time
import os

lcd = CharLCD(cols=16, rows=2, pin_rs=37, pin_e=35, pins_data=[33, 31, 29, 23], numbering_mode=GPIO.BOARD)

# lcd.write_string(u'Chuuuuuuuuuj!')
# lcd.crlf()
# lcd.write_string(u'PogChomp')

framebuffer = [
    'Ziobro!',
    '',
]

def write_to_lcd(lcd, framebuffer, num_cols):
    # Write the framebuffer out to the specifed LCD.
    lcd.home()
    for row in framebuffer:
        lcd.write_string(row.ljust(num_cols)[:num_cols])
        lcd.write_string('\r\n')

# write_to_lcd(lcd, framebuffer, 16)

long_string = 'Ty kurwo przestan mi wreszcie rodzine przesladowac'
def loop_string(string, lcd, framebuffer, row, num_cols, delay=0.25):
    padding = ' ' * num_cols
    s = padding + string + padding
    for i in range(len(s) - num_cols + 1):
        framebuffer[row] = s[i:i+num_cols]
        write_to_lcd(lcd, framebuffer, num_cols)
        time.sleep(delay)

if __name__ == '__main__':
    try:
        while True:
            f = os.popen("/opt/vc/bin/vcgencmd measure_temp")
            temp = f.read()
            temp_u = unicode(temp, 'utf-8')
            # loop_string(long_string, lcd, framebuffer, 1, 16)
            lcd.write_string(temp_u)
            time.sleep(2)
            lcd.clear()
    except KeyboardInterrupt:
        print("Measurement stooped by User")
        GPIO.cleanup()