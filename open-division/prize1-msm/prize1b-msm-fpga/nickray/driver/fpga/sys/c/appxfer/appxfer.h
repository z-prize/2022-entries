#ifndef APPXFER
#define APPXFER

#include <stdint.h>

int init_file(char* fn_app_to_sim, char* fn_sim_to_app);
int init_sock(const char*, const char*);
int init_f1(int, int);

void write_32_f1(uint32_t add, uint32_t data);

void write_512_f1(uint64_t off, void* buf);
void write_512_sock(uint64_t off, void* buf);
void write_flush();

void write_32x16_f1(uint64_t off, int i, uint32_t n);

uint32_t read_32_f1(uint32_t addr);

volatile uint32_t*      dma_wait_512(void);

void cmd_send(void* buf, uint32_t flush);

void pt_send(uint32_t pi, void* buf);

#endif
