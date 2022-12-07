/***

Copyright (c) 2022, Yrrid Software, Inc.  All rights reserved.
Licensed under the Apache License, Version 2.0, see LICENSE for details.

Author(s):  Niall Emmart

***/

bool parseHex(uint8_t* buffer, FILE* file, int bytes) {
  int next=0, count, nibble, current;
  int nibbles[bytes*2];
  
  for(int i=0;i<bytes;i++) {
    buffer[i]=0;
    nibbles[2*i+0]=0;
    nibbles[2*i+1]=0;
  }

  current=getc(file);
  while(current==' ' || current=='\r' || current=='\n' || current=='\t' || current==-1) {
    if(current==-1)
      return false;
    current=getc(file);
  }
    
  for(int i=0;i<bytes*2;i++) {
    if(current>='0' && current<='9') 
      nibble=current-'0';
    else if(current>='a' && current<='f')
      nibble=current-'a'+10;
    else if(current>='A' && current<='F')
      nibble=current-'A'+10;
    else
      break;
    nibbles[next++]=nibble;    
    current=getc(file);
  }
  
  if(current!='\r' && current!='\n' && current!='\t' && current!=' ' && current!=-1) {
    printf("Bad hex value in input file\n");
    exit(1);
  }
  
  count=next;
  next=0;
  for(int i=0;i<count;i++) {
    if(i%2==0)
      buffer[i/2]+=nibbles[count-1-i];
    else
      buffer[i/2]+=nibbles[count-1-i]*16;
  }
  return true;
}
