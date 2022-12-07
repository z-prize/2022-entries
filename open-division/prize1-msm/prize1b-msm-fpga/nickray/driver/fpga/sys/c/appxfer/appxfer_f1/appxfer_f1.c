
#include <stdio.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <immintrin.h>
#include <dirent.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>
#include <stdint.h>
#include <stddef.h>
#include <pthread.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <stdarg.h>
#include <sys/mman.h>
#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdarg.h>
#include <assert.h>
#include <string.h>
#include <sys/mman.h>
#include <immintrin.h>

#include <fpga_pci.h>
#include <fpga_mgmt.h>
#include <utils/lcd.h>

static uint16_t pci_vendor_id = 0x1D0F; /* Amazon PCI Vendor ID */
static uint16_t pci_device_id = 0xF001; /* PCI Device ID preassigned by Amazon for F1 applications */

static uint32_t pci_offset;
static pci_bar_handle_t pci_bar_handle_0;
static pci_bar_handle_t pci_bar_handle_4;
volatile uint32_t* pcie_addr_4;
volatile uint32_t* dma_buf;
static uint32_t* st_buf;

uint32_t cmd_off;
uint32_t cmd_base;


uint64_t m_mmap_dma(int);
uint32_t read_32_f1(uint32_t);
void write_32_f1(uint32_t, uint32_t);

extern void dma_set_seq(uint32_t);

int check_afi_ready(int slot_id) {
   struct fpga_mgmt_image_info info = {0};
   int rc;

   /* get local image description, contains status, vendor id, and device id. */
   rc = fpga_mgmt_describe_local_image(slot_id, &info,0);
   fail_on(rc, out, "Unable to get AFI information from slot %d. Are you running as root?",slot_id);

   /* check to see if the slot is ready */
   if (info.status != FPGA_STATUS_LOADED) {
     rc = 1;
     fail_on(rc, out, "AFI in Slot %d is not in READY state !", slot_id);
   }
/*
   printf("AFI PCI  Vendor ID: 0x%x, Device ID 0x%x\n",
          info.spec.map[FPGA_APP_PF].vendor_id,
          info.spec.map[FPGA_APP_PF].device_id);
*/
   /* confirm that the AFI that we expect is in fact loaded */
   if (info.spec.map[FPGA_APP_PF].vendor_id != pci_vendor_id ||
       info.spec.map[FPGA_APP_PF].device_id != pci_device_id) {
     printf("AFI does not show expected PCI vendor id and device ID. If the AFI "
            "was just loaded, it might need a rescan. Rescanning now.\n");

     rc = fpga_pci_rescan_slot_app_pfs(slot_id);
     fail_on(rc, out, "Unable to update PF for slot %d",slot_id);
     /* get local image description, contains status, vendor id, and device id. */
     rc = fpga_mgmt_describe_local_image(slot_id, &info,0);
     fail_on(rc, out, "Unable to get AFI information from slot %d",slot_id);

     printf("AFI PCI  Vendor ID: 0x%x, Device ID 0x%x\n",
            info.spec.map[FPGA_APP_PF].vendor_id,
            info.spec.map[FPGA_APP_PF].device_id);

     /* confirm that the AFI that we expect is in fact loaded after rescan */
     if (info.spec.map[FPGA_APP_PF].vendor_id != pci_vendor_id ||
         info.spec.map[FPGA_APP_PF].device_id != pci_device_id) {
       rc = 1;
       fail_on(rc, out, "The PCI vendor id and device of the loaded AFI are not "
               "the expected values.");
     }
   }

   return rc;
out:
   return 1;
}

int init_f1(int slot_id, int off)
{
    int rc;
    pci_bar_handle_0 = PCI_BAR_HANDLE_INIT;
    pci_bar_handle_4 = PCI_BAR_HANDLE_INIT;
    pci_offset = off;

    rc = fpga_mgmt_init();
    fail_on(rc, out, "Unable to initialize the fpga_mgmt library");

    rc = check_afi_ready(slot_id);
    fail_on(rc, out, "AFI not ready");

    rc = fpga_pci_attach(slot_id, FPGA_APP_PF, APP_PF_BAR0, 0, &pci_bar_handle_0);
    fail_on(rc, out, "Unable to attach to the AFI on slot id %d", slot_id);
    rc = fpga_pci_attach(slot_id, FPGA_APP_PF, APP_PF_BAR4, BURST_CAPABLE, &pci_bar_handle_4);
    fail_on(rc, out, "Unable to attach to the AFI on slot id %d", slot_id);

    fpga_pci_get_address(pci_bar_handle_4, 0, 1024*1024, (void**)&pcie_addr_4);

    st_buf = mmap(0, 512/8, PROT_READ|PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);

    uint64_t dma_phys = m_mmap_dma(4*1024);
    uint32_t coff = 0;
    write_32_f1(coff + (0x1e<<2), (dma_phys >>  0) & 0xffffffff);
    write_32_f1(coff + (0x1f<<2), (dma_phys >> 32) & 0xffffffff);
    dma_set_seq(read_32_f1(coff + (0x11<<2)));

    cmd_base = 0;
    cmd_off = 0;
    
    return rc;
out:
    if (pci_bar_handle_0 >= 0) {
        rc = fpga_pci_detach(pci_bar_handle_0);
        if (rc) {
            printf("Failure while detaching from the fpga.\n");
        }
    }
    return (rc != 0 ? 1 : 0);
}

uint32_t read_32_f1(uint32_t addr)
{
    int rc;
    uint32_t value;
    addr |= (1L << 30L);
    fpga_pci_poke(pci_bar_handle_0, pci_offset+0, addr);
    rc = fpga_pci_peek(pci_bar_handle_0, pci_offset+0, &value);
    fail_on(rc, out, "Unable to read from the fpga !");
out:
    return value;
}

void write_32_f1(uint32_t addr, uint32_t v)
{
    addr |= (2L << 30L);
    fpga_pci_poke(pci_bar_handle_0, pci_offset+4, v);
    fpga_pci_poke(pci_bar_handle_0, pci_offset+0, addr);
}

void write_256(uint64_t off, void* buf)
{
    uint32_t* data = (uint32_t*)buf;
    volatile uint32_t* addr = pcie_addr_4;
    addr += (off >> 2);
    int i = 0;
    if (0)
    {
        addr[i+0] = data[i+0];
        addr[i+1] = data[i+1];
        addr[i+2] = data[i+2];
        addr[i+3] = data[i+3];
        addr[i+4] = data[i+4];
        addr[i+5] = data[i+5];
        addr[i+6] = data[i+6];
        addr[i+7] = data[i+7];
    } else
    {
        __m256i v;
        v = _mm256_load_si256((__m256i*)data);
        _mm256_stream_si256((__m256i*)(addr), v);
        // __m256i v;
        // v = _mm256_load_si256((__m256i*)&data[i]);
        // _mm256_stream_si256((__m256i*)(&addr[i]), v);
    }
}

void write_512_f1(uint64_t off, uint8_t* buf)
{
    write_256(off+ 0, &buf[ 0]);
    write_256(off+32, &buf[32]);
}

void write_32x16_f1(uint64_t off, int i, uint32_t n)
{
    st_buf[i] = n;
    if (i == 16-1)
    {
        write_256(off+ 0, &st_buf[0]);
        write_256(off+32, &st_buf[8]);
    }
    _mm_sfence();
}

void write_flush()
{
    _mm_sfence();
}






















typedef struct {
    uint64_t pfn : 55;
    unsigned int soft_dirty : 1;
    unsigned int file_page : 1;
    unsigned int swapped : 1;
    unsigned int present : 1;
} PagemapEntry;

int pagemap_get_entry(PagemapEntry *entry, int pagemap_fd, uintptr_t vaddr)
{
    size_t nread;
    ssize_t ret;
    uint64_t data;
    uintptr_t vpn;

    vpn = vaddr / sysconf(_SC_PAGE_SIZE);
    nread = 0;
    while (nread < sizeof(data)) {
        ret = pread(pagemap_fd, &data, sizeof(data) - nread,
                vpn * sizeof(data) + nread);
        nread += ret;
        if (ret <= 0) {
            return 1;
        }
    }
    entry->pfn = data & (((uint64_t)1 << 55) - 1);
    entry->soft_dirty = (data >> 55) & 1;
    entry->file_page = (data >> 61) & 1;
    entry->swapped = (data >> 62) & 1;
    entry->present = (data >> 63) & 1;
    return 0;
}

int virt_to_phys_user(uintptr_t *paddr, pid_t pid, uintptr_t vaddr)
{
    char pagemap_file[BUFSIZ];
    int pagemap_fd;

    snprintf(pagemap_file, sizeof(pagemap_file), "/proc/%ju/pagemap", (uintmax_t)pid);
    pagemap_fd = open(pagemap_file, O_RDONLY);
    if (pagemap_fd < 0) {
        return 1;
    }
    PagemapEntry entry;
    if (pagemap_get_entry(&entry, pagemap_fd, vaddr)) {
        return 1;
    }
    close(pagemap_fd);
    *paddr = (entry.pfn * sysconf(_SC_PAGE_SIZE)) + (vaddr % sysconf(_SC_PAGE_SIZE));
    return 0;
}

uint64_t m_mmap_dma(int size)
{
    int r;

    if (dma_buf == 0)
    {
        dma_buf = mmap(0, size, PROT_READ|PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_LOCKED, -1, 0);
        //dma_buf = mmap(0, size, PROT_READ|PROT_WRITE, MAP_SHARED | MAP_LOCKED, -1, 0);
        if (dma_buf == 0)
        {
            printf ("error-mmap: %d\n", errno);
            return 1;
        }
        r = mlock((void*)dma_buf, size);
        if (r)
        {
            printf ("error-mlock: %d\n", errno);
            return 1;
        }
        for (int i = 0; i < size/4; i ++)
            dma_buf[i] = 0;
    }
    uintptr_t p;
    if (virt_to_phys_user(&p, getpid(), (uintptr_t)dma_buf))
    {
        printf ("error\n");
        return 1;
    }
    printf ("dma: %p -> %lx\n", dma_buf, p);

    return p;
}

