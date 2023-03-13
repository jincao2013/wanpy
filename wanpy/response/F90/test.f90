!module test
!
!    contains
!
!    function inv(Amat,ndim)
!        implicit none
!
!        integer,parameter    :: dp=8
!        integer              :: i
!        integer              :: info
!        integer,intent(in)   :: ndim
!
!        integer,allocatable       :: ipiv(:)
!
!        ! complex(dp)               :: ive(:,:)
!        complex(dp),parameter     :: zone=(1.0d0,0.0d0)
!        complex(dp),intent(in)    :: Amat(ndim,ndim)
!        complex(dp)               :: Bmat(ndim,ndim)
!        complex(dp),dimension(ndim,ndim)   :: inv
!
!        allocate(ipiv(ndim))
!        ipiv=0
!
!        ! unit matrix
!        Bmat= (0d0, 0d0)
!        do i=1,ndim
!            Bmat(i,i)= zone
!        enddo
!
!        call zgesv(ndim,ndim,Amat,ndim,ipiv,Bmat,ndim,info)
!
!        ! if(info .ne. 0)print *,'something wrong with zgesv'
!
!        inv = Bmat
!
!    end function inv
!
!    subroutine passArray(A,B,ndim)
!        implicit none
!
!        integer,parameter    :: dp=8
!        integer              :: i
!        integer,intent(in)   :: ndim
!        complex(dp),intent(in)    :: A(ndim,ndim)
!        complex(dp),intent(out)   :: B(ndim,ndim)
!        !f2py intent(in)  A
!        !f2py intent(in)  ndim
!        !f2py intent(out) B
!
!        B = A + A
!
!    end subroutine passArray
!
!end module test

program main
    ! implicit none
    
    use test

    integer,parameter    :: dp=8
    integer,parameter    :: ndim=2
    integer              :: i,j
    complex(dp)          :: a(ndim,ndim), b(ndim,ndim)
    ! complex(dp),dimension(ndim,ndim),external   :: inv

    a = 0
    a(1,1) = 1.0
    a(2,2) = 2.0
    ! b = inv(2,a)
    call passArray(a,b,2)
    
    print *, a(1,1), a(1,2)
    print *, a(2,1), a(2,2)
    print *, ''
    print *, b(1,1), b(1,2)
    print *, b(2,1), b(2,2)

end program main


