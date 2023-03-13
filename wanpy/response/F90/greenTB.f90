! *************************************
!  GreenTB Module
!
!  Coded by JinCao Sep.5, 2018
!
! !f2py --fcompiler=intelem --compiler=intelem --f90flags='-fpp' --opt='-fast' -L$MKLROOT/lib/intel64/ -lmkl_rt -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -liomp5 -c greenTB.f90 -m greenTB
! !f2py --fcompiler=intelem --compiler=intelem --f90flags='-fpp' -L$MKLROOT/lib/intel64/ -lmkl_rt -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -liomp5 -c greenTB.f90 -m greenTB
!
! *************************************

module greenTB

    contains

    function inv(A,ndim)
        implicit none

        integer,parameter    :: dp=8
        integer              :: i
        integer              :: info
        integer,intent(in)   :: ndim

        integer,allocatable       :: ipiv(:)

        ! complex(dp)               :: ive(:,:)
        complex(dp),parameter     :: zone=(1.0d0,0.0d0)
        complex(dp),intent(in)    :: A(ndim,ndim)
        complex(dp)               :: Amat(ndim,ndim),Bmat(ndim,ndim)
        complex(dp),dimension(ndim,ndim)   :: inv

        allocate(ipiv(ndim))
        ipiv=0

        ! unit matrix
        Bmat= (0d0, 0d0)
        do i=1,ndim
            Bmat(i,i)= zone
        enddo

        Amat = A
        call zgesv(ndim,ndim,Amat,ndim,ipiv,Bmat,ndim,info)

        ! if(info .ne. 0)print *,'something wrong with zgesv'

        inv = Bmat

    end function inv

    subroutine self_energy(hs,h0,h1,h2,selfen,nw)
        implicit none

        integer,parameter       :: dp=8

        integer                 :: N,i
        integer,intent(in)      :: nw
        real(dp), parameter     :: eps8 = 1.0e-8_dp
        complex(dp),parameter   :: zone=(1.0d0,0.0d0)
        complex(dp),intent(in)  :: hs(nw,nw),h0(nw,nw),h1(nw,nw),h2(nw,nw)
        complex(dp),intent(out) :: selfen(nw,nw)
        complex(dp)             :: g(nw,nw),gs(nw,nw),gs_new(nw,nw)
        complex(dp)             :: a(nw,nw),b(nw,nw),agb(nw,nw),bga(nw,nw)
        complex(dp)             :: ones(nw,nw)
        !f2py integer intent(in)                            nw
        !f2py complex intent(in),Dimension(nw,nw)           hs,h0,h1,h2
        !f2py complex intent(out),Dimension(nw,nw,ne)       selfen

        N = 8000
        do i=1,nw
            ones(i,i) = zone
        end do
        a = 0
        b = 0
        agb = 0
        bga = 0
        g = 0
        gs = 0
        gs_new = 0
        selfen = 0

        a = -h1
        b = -h2
        g = inv(h0, nw)
        gs = inv(hs, nw)

        do i=1,N
            agb = matmul(matmul(a, g), b)
            bga = matmul(matmul(b, g), a)
            gs_new = matmul(inv(ones - matmul(gs, agb), nw), gs)

            if (maxval(abs(gs_new - gs)) .lt. eps8) exit

            gs = gs_new
            b = matmul(matmul(b, g), b)
            a = matmul(matmul(a, g), a)
            g = matmul(inv(ones - matmul(g, agb) - matmul(g, bga), nw), g)

!            print *, 'iter=', i

            if (i .eq. N-1) print *, 'Warning ! iterative procedure not reach accuracy !'

        end do
        selfen = matmul(matmul(h1,gs),h2)

    end subroutine self_energy

    subroutine get_selfen(er,hs,h0,h1,h2,selfenLr,selfenRr,fermi,nw,ne)
        implicit none

        integer,parameter         :: dp=8
        integer                   :: i
        integer,intent(in)        :: nw,ne
        real(dp),intent(in)       :: fermi

        complex(dp),parameter     :: zone=(1.0d0,0.0d0)
        complex(dp)               :: ones(nw,nw)
        complex(dp),intent(in)    :: er(ne)
        complex(dp),intent(in)    :: hs(nw,nw),h0(nw,nw),h1(nw,nw),h2(nw,nw)
        complex(dp),intent(out)   :: selfenLr(nw,nw,ne),selfenRr(nw,nw,ne)
        complex(dp)               :: selfen(nw,nw),hss(nw,nw),h00(nw,nw)

        !f2py real intent(in)                             fermi
        !f2py integer intent(in)                          nw,ne
        !f2py complex intent(in),dimension(nw,nw)         hs,h0,h1,h2
        !f2py complex intent(out),dimension(nw,nw,ne)     selfenLr,selfenRr

        selfenLr = 0
        selfenRr = 0
        selfen = 0
        ones = 0
        hss = 0
        h00 = 0
        do i=1,nw
            ones(i,i) = zone
        end do
!        print *, '[In get_selfen] ones=', ones

        do i=1,ne
            hss = er(i)*ones-hs
            h00 = er(i)*ones-h0
            call self_energy(hss, h00, h1, h2, selfen, nw)
            selfenLr(:,:,i) = selfen
            call self_energy(hss, h00, h2, h1, selfen, nw)
            selfenRr(:,:,i) = selfen
        end do
    end subroutine get_selfen

    subroutine get_Gr(er,hs,h0,h1,h2,GBr,GLr,GRr,fermi,nw,ne)
        implicit none

        integer,parameter         :: dp=8
        integer                   :: i
        integer,intent(in)        :: nw,ne
        real(dp),intent(in)       :: fermi

        complex(dp),parameter     :: zone=(1.0d0,0.0d0)
        complex(dp)               :: ones(nw,nw)
        complex(dp),intent(in)    :: er(ne)
        complex(dp),intent(in)    :: hs(nw,nw),h0(nw,nw),h1(nw,nw),h2(nw,nw)
        complex(dp)               :: selfenLr(nw,nw,ne),selfenRr(nw,nw,ne)
        complex(dp),intent(out)   :: GBr(nw,nw,ne),GLr(nw,nw,ne),GRr(nw,nw,ne)

        !f2py intent(in)           nw,ne
        !f2py intent(in)           eps,fermi
        !f2py intent(in)           hs,h0,h1,h2
        !f2py intent(out)          GBr,GLr,GRr

        ones = 0
        GBr = 0
        GLr = 0
        GRr = 0
        do i=1,nw
            ones(i,i) = zone
        end do

        call get_selfen(er,hs,h0,h1,h2,selfenLr,selfenRr,fermi,nw,ne)

        do i=1,ne
            GBr(:,:,i) = inv(er(i)*ones-h0-selfenLr(:,:,i), nw)
            GLr(:,:,i) = inv(er(i)*ones-h0-selfenRr(:,:,i), nw)
            GRr(:,:,i) = inv(er(i)*ones-h0-selfenLr(:,:,i)-selfenRr(:,:,i), nw)
        end do
    end subroutine get_Gr

end module greenTB

!program main
!    ! implicit none
!
!    use greenTB
!
!    integer,parameter    :: dp=8
!    integer,parameter    :: nw=4,ne=5
!    integer              :: i,j
!    real(dp),parameter   :: eta=0.01,fermi=0.0
!    real(dp)             :: ee(ne)
!    complex(dp)          :: er(ne)
!    complex(dp)          :: hs(nw,nw),h0(nw,nw),h1(nw,nw),h2(nw,nw)
!    complex(dp)          :: selfen(nw,nw),e(nw,nw)
!    complex(dp)          :: selfenLr(nw,nw,ne),selfenRr(nw,nw,ne)
!    complex(dp)          :: GBr(nw,nw,ne),GLr(nw,nw,ne),GRr(nw,nw,ne)
!    ! complex(dp),dimension(ndim,ndim),external   :: inv
!
!
!    ee = (/ 1.1, 1.2, 1.3, 1.4, 1.5 /)
!    er = ee + eta * (0,1)
!    print *, er
!
!    e=0
!    do i=1,nw
!        e(i,i) = (0.1,eta)
!    end do
!
!!    h0 = 0
!!    h0(1,1) = (0.74471898,0.0)
!!    h0(2,2) = (0.84463567,0.0)
!!    h0(1,2) = (0.04991671,0.0)
!!    h0(2,1) = (0.04991671,0.0)
!!
!!    h0(1,1) = (1,0.0)
!!    h0(2,2) = (4,0.0)
!!    h0(1,2) = (2,0.0)
!!    h0(2,1) = (3,0.0)
!!    hs = h0
!!
!!    h1 = 0
!!    h2 = 0
!!    h1(1,1) = (2.50,0.0)
!!    h1(2,2) = (-2.50,0.0)
!!    h1(1,2) = (-0.25,0.0)
!!    h1(2,1) = (0.25,0.0)
!!    h2(1,1) = (2.50,0.0)
!!    h2(2,2) = (-2.50,0.0)
!!    h2(1,2) = (0.25,0.0)
!!    h2(2,1) = (-0.25,0.0)
!
!    hs = 0
!    h1 = 0
!    h2 = 0
!    h0 = 0
!    hs(1,:)=(/ ( 0.74471898,0),  ( 0.04991671,0),  ( 2.50000000,0), (-0.25000000,0) /)
!    hs(2,:)=(/ ( 0.04991671,0),  ( 0.84463567,0),  ( 0.25000000,0), (-2.50000000,0) /)
!    hs(3,:)=(/ ( 2.50000000,0),  ( 0.25000000,0),  ( 0.74471898,0), ( 0.04991671,0) /)
!    hs(4,:)=(/ (-0.25000000,0),  (-2.50000000,0),  ( 0.04991671,0), ( 0.84463567,0) /)
!    h0 = hs
!
!    h1(1,:) = (/ (0,0),  (0,0),  ( 2.5 ,0), ( 0.25,0) /)
!    h1(2,:) = (/ (0,0),  (0,0),  (-0.25,0), (-2.5 ,0) /)
!    h2(3,:) = (/ (2.50,0),  (-0.25,0),  (0,0),  (0,0) /)
!    h2(4,:) = (/ (0.25,0),  (-2.50,0),  (0,0),  (0,0) /)
!
!!    print *, hs(1,:)
!!    print *, hs(2,:)
!!    print *, hs(3,:)
!!    print *, hs(4,:)
!!    print *, h0
!!    print *, h1
!!    print *, h2
!!    hs = matmul(hs,h0)
!!    print *, hs(1,1), hs(1,2)
!!    print *, hs(2,1), hs(2,2)
!
!!   call self_energy(e-hs,e-h0,h1,h2,selfen,nw)
!!
!!    print *, selfen(1,1), selfen(1,2)
!!    print *, selfen(2,1), selfen(2,2)
!
!!    call get_selfen(er,hs,h0,h1,h2,selfenLr,selfenRr,fermi,nw,ne)
!
!!    call get_Gr(er,hs,h0,h1,h2,GBr,GLr,GRr,fermi,nw,ne)
!
!!    print *, 'selfenLr=\n', selfenLr
!!    print *, 'selfenRr=\n', selfenRr
!
!!    print *, a(1,1), a(1,2)
!!    print *, a(2,1), a(2,2)
!!    print *, ''
!!    print *, b(1,1), b(1,2)
!!    print *, b(2,1), b(2,2)
!
!end program main
