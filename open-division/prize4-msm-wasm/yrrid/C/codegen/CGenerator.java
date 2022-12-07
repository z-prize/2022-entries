/***

Copyright (c) 2022, Yrrid Software, Inc.  All rights reserved.
Licensed under the Apache License, Version 2.0, see LICENSE for details.

Author(s):  Niall Emmart

***/

import java.io.*;

public class CGenerator {
  static public final int[] N=new int[]{
      0x3fffaaab, 0x27fbffff, 0x153ffffb, 0x2affffac, 0x30f6241e, 0x034a83da,
      0x112bf673, 0x12e13ce1, 0x2cd76477, 0x1ed90d2e, 0x29a4b1ba, 0x3a8e5ff9, 
      0x001a0111
  };

  static public final int[] NP=new int[]{
      0x3ffcfffd, 0x27cffff3, 0x1d113e88, 0x1ab6e4b6, 0x230b4828, 0x3cbbc323,
      0x2db4c16e, 0x3283a3ac, 0x0f5819ec, 0x1bfb89a3, 0x09468b31, 0x01bfaabf,
      0x20ceb061
  };

  static public void mad(PrintWriter pw, int i, int j, int k) {
    pw.println("  p" + k + "=p" + k + " + (a->f" + i + ")*(b->f" + j + ");");
  }

  static public void madsq(PrintWriter pw, int i, int j, int k) {
    pw.println("  p" + k + "=p" + k + " + (a->f" + i + ")*(a->f" + j + ");");
  }

  static public void madsab(PrintWriter pw, int i, int j, int k) {
    pw.println("  sp" + k + "=sp" + k + " + sa" + i + "*sb" + j + ";");
  }

  static public void madq(PrintWriter pw, int j, int k) {
    pw.printf("  p" + k + "=p" + k + " + q*0x%08X;\n", N[j]);
  }

  static public void madklo(PrintWriter pw, int i, int j, int k, int[] constBN) {
    pw.printf("  r" + k + "=r" + k + " + (p" + i + "+p" + j + ")*(0x%08X + 0x%08X);\n", constBN[i], constBN[j]);
  }

  static public void madkhi(PrintWriter pw, int i, int j, int k, int[] constBN) {
    pw.printf("  p" + k + "=p" + k + " + (r" + i + "+r" + j + ")*(0x%08X + 0x%08X);\n", constBN[i], constBN[j]);
  }


  static public void gradeSchoolMultiply(PrintWriter pw) {
    for(int i=0;i<13;i++) {
      for(int j=0;j<13;j++)
        mad(pw, i, j, i+j);
    }
  }

  static public void fastSquareMultiply(PrintWriter pw) {
    for(int i=0;i<13;i++) {
      for(int j=0;j<i;j++)
        madsq(pw, i, j, i+j);
    }
    for(int i=0;i<25;i++) {
      if(i%2==0)
        pw.println("  p" + i + "=(p" + i + "<<1) + a->f" + (i/2) + "*a->f" + (i/2) + ";");
      else
        pw.println("  p" + i + "=p" + i + "<<1;");
    }
  }

  static public void karatsubaMultiply(PrintWriter pw) {
    pw.println("  uint64_t sa0, sa1, sa2, sa3, sa4, sa5, sa6, sb0, sb1, sb2, sb3, sb4, sb5, sb6;");
    pw.println("  uint64_t sp0, sp1, sp2, sp3, sp4, sp5, sp6, sp7, sp8, sp9, sp10, sp11, sp12;");
    pw.println();
    pw.println("  sp0=0; sp1=0; sp2=0; sp3=0; sp4=0; sp5=0; sp6=0; sp7=0; sp8=0; sp9=0; sp10=0; sp11=0; sp12=0;");
    pw.println();
    pw.println("  // +/- ops:  143");
    pw.println("  // * ops:    135");
    pw.println();

    for(int i=0;i<6;i++) {
      pw.println("  sa" + i + " = (a->f" + i + ") + (a->f" + (i+7) + ");");
      pw.println("  sb" + i + " = (b->f" + i + ") + (b->f" + (i+7) + ");");
    }
    pw.println("  sa6 = a->f6;");
    pw.println("  sb6 = b->f6;");

    for(int i=0;i<7;i++) {
      for(int j=0;j<7;j++) 
        mad(pw, i, j, i+j);
    }

    for(int i=0;i<7;i++) {
      for(int j=0;j<7;j++)
        madsab(pw, i, j, i+j);
    }

    for(int i=7;i<13;i++) {
      for(int j=7;j<13;j++) 
        mad(pw, i, j, i+j);
    }

    for(int i=0;i<11;i++) 
      pw.println("  sp" + i + " = sp" + i + " - p" + i + " - p" + (i+14) + ";");
    pw.println("  sp11 = sp11 - p11;");
    pw.println("  sp12 = sp12 - p12;");

    for(int i=0;i<13;i++)
      pw.println("  p" + (i+7) + " += sp" + i + ";");
  }

  static public void gradeSchoolReduce(PrintWriter pw) {
    pw.println("  p9 += p8>>30;");
    pw.println("  p10 += p9>>30;");
    pw.println("  p11 += p10>>30;");
    pw.println("  p12 += p11>>30;");
    pw.println("  p13 += p12>>30;");
    pw.println("  p14 += p13>>30;");
    pw.println("  p15 += p14>>30;");
    pw.println("  p16 += p15>>30;");
    pw.println("  p17 += p16>>30;");
    pw.println("  p8 = p8 & mask;");
    pw.println("  p9 = p9 & mask;");
    pw.println("  p10 = p10 & mask;");
    pw.println("  p11 = p11 & mask;");
    pw.println("  p12 = p12 & mask;");
    pw.println("  p13 = p13 & mask;");
    pw.println("  p14 = p14 & mask;");
    pw.println("  p15 = p15 & mask;");
    pw.println("  p16 = p16 & mask;");

    for(int i=0;i<13;i++) {
      pw.println("  q=p" + i + "*0x3FFCFFFD & mask;");
      for(int j=0;j<13;j++) 
        madq(pw, j, j+i);
      pw.println("  p" + (i+1) + "=p" + (i+1) + " + (p" + i + ">>30);");
    }
  }

  static public void karatsubaReduce(PrintWriter pw) {
    for(int i=1;i<18;i++) 
      pw.println("  p" + i + " += p" + (i-1) + ">>30;");
    for(int i=0;i<17;i++)
      pw.println("  p" + i + " = p" + i + " & mask;");

    pw.println("  uint64_t d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12;");
    pw.println("  uint64_t r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13;");

    for(int i=0;i<13;i++)
      pw.printf("  d" + i + " = p" + i + "*0x%08X;\n", NP[i]);
  
    pw.println("  r0 = d0;");
    pw.println("  r1 = -d0-d1;");
    for(int i=2;i<13;i++)
      pw.println("  r" + i + " = r" + (i-1) + " - d" + i + ";");
    for(int i=1;i<7;i++)
      pw.println("  r" + (2*i) + " = r" + (2*i) + " + (d" + i + "<<1);");

    for(int i=0;i<13;i++)
      for(int j=0;j<13;j++)
        if(i<j && i+j<13)
          madklo(pw, i, j, i+j, NP);

    for(int i=1;i<14;i++) 
      pw.println("  r" + i + " += r" + (i-1) + ">>30;");
    for(int i=0;i<13;i++)
      pw.println("  r" + i + " = r" + i + " & mask;");

    for(int i=0;i<13;i++)
      for(int j=0;j<13;j++)
        if(i<j && i+j>=11)
          madkhi(pw, i, j, i+j, N);

    for(int i=0;i<13;i++)
      pw.printf("  d" + i + " = r" + i + "*0x%08X;\n", N[i]);

    pw.println("  r13 = d12;");
    pw.println("  r12 = -d11-d12;");
    for(int i=11;i>=1;i--)
      pw.println("  r" + i + " = r" + (i+1) + " - d" + (i-1) + ";"); 
    pw.println("  r0 = r1 + d12;");

    for(int i=0;i<6;i++)
      pw.println("  r" + (2*i+1) + " = r" + (2*i+1) + " + (d" + (i+6) + "<<1);");

    for(int i=0;i<14;i++)
      pw.println("  p" + (i+11) + " += r" + i + ";");

    pw.println("  p12 += 256 + (p11>>30);");
    pw.println("  p13 += p12>>30;");
  }

  static public void noResolve(PrintWriter pw) {
    for(int i=0;i<12;i++) 
      pw.println("  r->f" + i + "=p" + (i+13) + ";");
    pw.println("  r->f12=0;");    
  }

  static public void resolve(PrintWriter pw) {
    for(int i=0;i<11;i++) {
      pw.println("  r->f" + i + "=p" + (i+13) + " & mask;");
      pw.println("  p" + (i+14) + "=p" + (i+14) + " + (p" + (i+13) + ">>30);");
    }
    pw.println("  r->f11=p24 & 0x3FFFFFFFu;");
    pw.println("  r->f12=p24>>30;");
  }

  static public void main(String[] args) throws IOException {
    PrintWriter gsmPW=new PrintWriter("../generated/grade_school_mult_v1.c");
    PrintWriter fsPW=new PrintWriter("../generated/fast_square_mult_v1.c");
    PrintWriter kmPW=new PrintWriter("../generated/karatsuba_mult_v1.c");
    PrintWriter gsrPW=new PrintWriter("../generated/grade_school_red_v1.c");
    PrintWriter krPW=new PrintWriter("../generated/karatsuba_red_v1.c");
    PrintWriter nrPW=new PrintWriter("../generated/no_resolve.c");
    PrintWriter rPW=new PrintWriter("../generated/resolve.c");

    gradeSchoolMultiply(gsmPW);
    fastSquareMultiply(fsPW);
    karatsubaMultiply(kmPW);
    gradeSchoolReduce(gsrPW);
    karatsubaReduce(krPW);
    noResolve(nrPW);
    resolve(rPW);

    gsmPW.close();
    fsPW.close();
    kmPW.close();
    gsrPW.close();
    krPW.close();
    nrPW.close();
    rPW.close(); 
  }
}
