
#include <stdio.h>

#include "appxfer.h"









uint32_t dma_off;
uint32_t dma_seq;

extern volatile uint32_t* dma_buf;
extern uint32_t cmd_off;
extern uint32_t cmd_base;

extern void write_flush();


extern uint32_t read_32_f1(uint32_t);









void dma_set_seq(uint32_t s)
{
    dma_seq = s;
    dma_off = (dma_seq & 0x3f) << 6;
}

uint32_t dma_read_32(uint32_t off)
{
    return dma_buf[off>>2];
}

void dma_wait_256(uint32_t off)
{
    uint32_t s = 0;
    do {
        s = dma_read_32(off);
    } while (s != dma_seq);
}

volatile uint32_t* dma_wait_512(void)
{
    volatile uint32_t* p = &dma_buf[dma_off>>2];
    dma_wait_256(dma_off);
    dma_wait_256(dma_off+32);
    // for (int i = 0; i < 7; i ++)
    //     s[i] = dma_read_32(dma_off+4+(i*4));
    // for (int i = 0; i < 7; i ++)
    //     s[7+i] = dma_read_32(dma_off+32+4+(i*4));
    dma_set_seq(dma_seq+1);
    return p;
}















void cmd_send(void* buf, uint32_t flush)
{
    // write_512(cmd_base + cmd_off, buf);
    // cmd_off += 1;
    // cmd_off &= 0x1f;
    // if (flush || (cmd_off == 0))
    //     write_flush();
}
















void pt_send(uint32_t pi, void* buf)
{
    // uint8_t* data = (uint8_t*)buf;
    // write_512(pi +   0, &data[  0]);
    // write_512(pi +  64, &data[ 64]);
    // write_512(pi + 128, &data[128]);
}
















// int main(int argc, char* argv[])
// {
//     printf ("main\n");
// }

