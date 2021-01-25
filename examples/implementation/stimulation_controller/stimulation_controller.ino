

elapsedMillis clock_time;


const byte clock_pin = 29;
const byte sync_pin = LED_BUILTIN;
const byte stim_pin = 3;

byte recv_byte;


unsigned long last_reset = 0;
unsigned long cycle_length = 0;
unsigned long current_phase;
unsigned long stim_start;
unsigned long stim_end;

byte wraparound = 0;

unsigned long stim_start_0;
unsigned long stim_end_0;
unsigned long cycle_length_0 = 0;

unsigned long isi = 5;
unsigned long last_trig = 1000;
unsigned long on_time = 10;
bool stim_on = false;


unsigned long read_uint32() {
  byte uint32_buffer = 0;
  const unsigned long factors[4] = {1, 256, 65536, 16777216};
  // big end first!!
  unsigned long value = 0;
  for(int i = 0; i<4; i++) {
    while (!Serial.available()) {
      ;
    }
    uint32_buffer = Serial.read();
    
    value += uint32_buffer * factors[i];
  }

  return value;
}


void reset_clock() {
  clock_time = 0;
  digitalWrite(clock_pin, HIGH);
  delay(1);
  digitalWrite(clock_pin, LOW);
}

void sync_clock() {
  unsigned long sent_time = 0;
  signed long time_diff;
  sent_time = clock_time;
  digitalWrite(clock_pin, HIGH);
  delayMicroseconds(500);
  digitalWrite(clock_pin, LOW);
  while (!Serial.available()) {
    ;
  }
  unsigned long recv_time = read_uint32();
  time_diff = recv_time - sent_time;
  clock_time += time_diff;
  last_trig += time_diff;
}

void predict_phase() {
  current_phase = clock_time - last_reset; 
  current_phase %= cycle_length;
}

void get_stimulation_params() {
  last_reset = read_uint32();
  cycle_length = read_uint32();
  stim_start = read_uint32();
  stim_end = read_uint32();
  if (stim_start < stim_end) {
    wraparound = 0;
  } else {
    wraparound = 1;
  }
}

bool do_stim = false;

void turn_on_stim() {
  digitalWrite(stim_pin, HIGH);
  digitalWrite(sync_pin, HIGH);
  do_stim = true;
}

void turn_off_stim() {
  digitalWrite(stim_pin, LOW);
  digitalWrite(sync_pin, LOW);
  do_stim = false;
}

void check_stimulus() {
  if (wraparound) {
    if ( (current_phase >= stim_start) || (current_phase < stim_end)  ) {
      turn_on_stim();
    } else {
      turn_off_stim();
    }
  } else {
    if ( (current_phase >= stim_start) && (current_phase < stim_end)  ) {
      turn_on_stim();
    } else {
      turn_off_stim();
    }
  }
}



void stimulate() {
  if (!stim_on and do_stim) {
    if (clock_time - last_trig >= isi) {
      digitalWrite(stim_pin, HIGH);
      digitalWrite(sync_pin, HIGH);
      last_trig = clock_time;
      stim_on = true;
    }
  }
  
  if (stim_on) {
    if (clock_time > (last_trig + on_time)) {
      digitalWrite(stim_pin, LOW);
      digitalWrite(sync_pin, LOW);
      stim_on = false;
    }
    
  }
}


void setup() {
  pinMode(clock_pin, OUTPUT);
  pinMode(stim_pin, OUTPUT);
  pinMode(sync_pin, OUTPUT);
  Serial.begin(115200);
}

byte run_loop = 0;

void loop() {
  
  run_loop = 0;
  
  while (!run_loop) {
    
    if (Serial.available()) {
      recv_byte = Serial.read();
    
    
      if (recv_byte== 255) {
        reset_clock();
        Serial.clear();
      }
      if (recv_byte == 254) {
        get_stimulation_params();
        run_loop = 1;
      }
    }
    
  }

  Serial.clear();
  digitalWrite(LED_BUILTIN, HIGH);
  while (run_loop) {
    
    if (Serial.available()) {
      recv_byte = Serial.read();
      if (recv_byte == 253) {
        get_stimulation_params();
      } else if (recv_byte == 244) {
        sync_clock();
      } else if (recv_byte == 200) {
        run_loop = 0;
        turn_off_stim();
        break;
      }
      Serial.clear();
    }
    
    predict_phase();
    check_stimulus();
    //stimulate();
    
  }
}
