/***

Copyright (c) 2022, Yrrid Software, Inc.  All rights reserved.
Licensed under the Apache License, Version 2.0, see LICENSE for details.

Author(s):  Niall Emmart
            Sougata Bhattacharya
            Antony Suresh

***/

#if defined(C_BUILD)
  #include <stdio.h>
  #include <stdlib.h>
#endif

char nibbles[16]={'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F'};

uint32_t logLength=0;
uint8_t* logBuffer=(void*)0;

void logInitialize(void* buffer) {
  #if defined(WASM_BUILD)
    logBuffer=(uint8_t*)buffer;
    logLength=0;
    logBuffer[0]=0;
  #endif
}

void logString(const char *str) {
  #if defined(C_BUILD)
    printf("%s", str);
  #else
    while(*str!=0) 
      logBuffer[logLength++]=*str++;
    logBuffer[logLength]=0;
  #endif 
}

void logHex(uint32_t value) {
  char hex[9];

  for(int i=7;i>=0;i--) {
    hex[i]=nibbles[value & 0x0F];
    value=value>>4;
  }
  hex[8]=0;
  logString(hex);
}

void logHex64(uint64_t value) {
  char hex[17];

  for(int i=15;i>=0;i--) {
    hex[i]=nibbles[value & 0x0F];
    value=value>>4;
  }
  hex[16]=0;
  logString(hex);
}

void logDec(int32_t value) {
  char dec[10];
  int  i;

  if(value<0) {
    logString("-");
    value=-value;
  }
  for(i=8;i>=0;i--) {
    dec[i]=nibbles[value % 10];
    value=value/10;
    if(value==0)
      break;
  }
  dec[9]=0;
  logString(dec+i);
}
